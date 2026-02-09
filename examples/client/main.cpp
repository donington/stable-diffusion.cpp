
// sd-client: main.cpp
#include <filesystem>
#include <fstream>
#include <format>
#include <future>
#include <atomic>

#include "httplib.h"
#include "stable-diffusion.h"

#include "common/common.hpp"
#include "common/base64.hpp"

namespace fs = std::filesystem;


/** helper struct for managing filename, extension, and loading image data **/
class file_data {
private:
    std::string v_filename;
    std::string v_extension;

    bool prepare(std::string const& filename_in, std::vector<std::string> const& valid_ext) {
        if (filename_in.size() < 1) {
            LOG_ERROR("error: empty file name", filename_in.c_str());
            return false;
        }

        auto pos = filename_in.rfind(".");
        std::string ext;

        if (pos != std::string::npos && pos != 0)
            ext = filename_in.substr(pos+1);

        if (ext == "jpg" || ext == "jpe") ext = "jpeg";

        for (auto const& extchk : valid_ext) {
            if (ext == extchk) {
                v_filename = filename_in;
                v_extension = ext;
                return true;
            }
        }

        LOG_ERROR("error: `%s`: unsupported file type", filename_in.c_str());
        return false;
    }

public:
    std::string const& filename() const {
        return v_filename;
    }

    std::string const& extension() const {
        return v_extension;
    }

    bool prepare_load(std::string const& filename_in, std::vector<std::string> const& valid_ext) {
        if (!prepare(filename_in, valid_ext))
            return false;
        if (!fs::exists(filename_in)) {
            LOG_ERROR("io error: `%s`: file not found", filename_in.c_str());
            return false;
        }
        return true;
    }

    bool prepare_save(std::string const& filename_in, std::vector<std::string> const& valid_ext, bool allow_overwrite = false) {
        if (!prepare(filename_in, valid_ext))
            return false;
        if (!allow_overwrite && fs::exists(filename_in)) {
            LOG_ERROR("io error: `%s`: file exists", filename_in.c_str());
            return false;
        }
        return true;
    }

    std::unique_ptr<std::vector<uint8_t>> load() const {
        std::vector<uint8_t> bytes;

        try {
            std::ifstream ifs;
            ifs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            ifs.open(v_filename);

            if (ifs.is_open()) {
                std::for_each(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>(), [&bytes](const char c) {
                    bytes.push_back(c);
                });
            }
            else {
                LOG_ERROR("io error: `%s`: failed opening file", v_filename.c_str());
                return std::unique_ptr<std::vector<uint8_t>>();
            }

            ifs.close();

        } catch (std::ios_base::failure const& e) {
            LOG_ERROR("io error: %s", e.what());
            return std::unique_ptr<std::vector<uint8_t>>();
        } catch (std::exception const& e) {
            LOG_ERROR("unexpected error: %s", e.what());
            return std::unique_ptr<std::vector<uint8_t>>();
        }

        return std::make_unique<std::vector<uint8_t>>(bytes);
    }

    bool save(std::vector<uint8_t> bytes) const {
        try {
            std::ofstream ofs;
            ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            ofs.open(v_filename);

            if (ofs.is_open()) {
                ofs.write(reinterpret_cast<char const*>(bytes.data()), bytes.size());
            }
            else {
                LOG_ERROR("io error: `%s`: failed opening file", v_filename.c_str());
                return false;
            }

            ofs.close();

        } catch (std::ios_base::failure const& e) {
            LOG_ERROR("io error: %s", e.what());
            return false;
        } catch (std::exception const& e) {
            LOG_ERROR("unexpected error: %s", e.what());
            return false;
        }

        return true;
    }
};

enum class ServerEndpoint {
    generate = 1,
    edit,
    unset = 0
};


struct SDClientParams {
    std::string  server_ip    = "";  // required
    int          server_port  = 1234;  // could set this via server_ip:port format

    std::string size    = "512x512";
    std::string format  = "png";  // could be determined from output file name (if specified)
    int compression = 100;

    std::string output_path = "";
    std::string output_file = "";
    int output_begin_idx;
    bool overwrite = false;

    std::vector<file_data> edit_images;
    file_data edit_mask;

    ServerEndpoint mode = ServerEndpoint::unset;  // required (-g|-e)
    SDMode sdmode = IMG_GEN;  // can probably be determined at runtime, somewhat (ignore for now)

    bool verbose  = false;
    bool color    = false;
    bool help     = false;
    bool version  = false;

    const std::vector<std::string> valid_image_types{"png","jpeg"};
    const std::vector<std::string> valid_file_types{"png","jpeg"};


    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"", "--server",
             "server to connect to (required)",
             &server_ip},
            {"-s", "--size",
             "image resolution (default: 512x512)",
             &size},
            {"-f", "--format",
             "output format [png, jpeg] (default: png)",
             &format},
            {"-O", "--output-path",
             "output path prefix for output file (default: unset)",
             &output_path},
            {"-o", "--output",
             "output file (default: tries to write to 'output-??.ext' (ext := format)",
             &output_file},
        };

        options.int_options = {
            {"", "--port",
             "server port (default: 1234)",
             &server_port},
            {"", "--compression",
             "server port (default: 100 (max compression))",
             &compression},
/* not yet ready for this flag
            {"--output-begin-idx",
             "starting index for output image sequence, must be non-negative (default 0 if specified %d in output path, 1 otherwise)",
             &output_begin_idx},
*/
        };

        options.bool_options = {
            {"", "--overwrite",
             "overwrite output file (if it already exists)",
             true, &overwrite},
            {"-v", "--verbose",
             "print extra info",
             true, &verbose},
            {"", "--color",
             "colors the logging tags according to level",
             true, &color},
        };

        auto on_endpoint_arg = [&](int argc, const char** argv, int index) {
            if (mode != ServerEndpoint::unset) {
                LOG_ERROR("error: multiple endpoints selected");
                return -1;
            }
            std::string flag{argv[index]};
            if (flag == "-g" || flag == "--generate")
                mode = ServerEndpoint::generate;
            if (flag == "-e" || flag == "--edit")
                mode = ServerEndpoint::edit;
            return 0;
        };

        auto on_image_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg  = argv[index];
            file_data data;
            if (!data.prepare_load(arg, valid_image_types))
                return -1;
            edit_images.push_back(data);
            return 1;
        };

        auto on_mask_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg  = argv[index];
            if (!edit_mask.prepare_load(arg, valid_image_types))
                return -1;
            return 1;
        };

        auto on_help_arg = [&](int argc, const char** argv, int index) {
            help = true;
            return 0;
        };

        auto on_version_arg = [&](int argc, const char** argv, int index) {
            help = true;
            return 0;
        };

        options.manual_options = {
            {"-g", "--generate",
             "use the server /v1/images/generations endpoint",
             on_endpoint_arg},
            {"-e", "--edit",
             "use the server /v1/images/edits endpoint",
             on_endpoint_arg},
            {"-i", "--image",
             "edit image input file(s) (can be specified more than once)",
             on_image_arg},
            {"-m", "--mask",
             "edit image mask input file (default: unset)",
             on_mask_arg},
            {"-h", "--help",
             "show this help message and exit",
             on_help_arg},
            {"", "--version",
             "print version and exit",
             on_version_arg},
        };
        return options;
    };

    bool process_and_check() {
        if (server_ip.empty()) {
            LOG_ERROR("error: server is required (--server HOST)");
            return false;
        }

        if (server_port < 0 || server_port > 65535) {
            LOG_ERROR("error: port should be in the range [0, 65535]");
            return false;
        }

        if (!size.empty()) {  // validate size parameter
            int width, height;
            auto pos = size.find('x');
            if (pos != std::string::npos) {
                try {
                    width  = std::stoi(size.substr(0, pos));
                    height = std::stoi(size.substr(pos + 1));
                } catch (...) {}
            }
            else {
                LOG_ERROR("error: invalid size specified (try something like --size 512x512)");
                return false;
            }
            if (width < 1 || height < 1) {
                LOG_ERROR("error: size must be a positive number (eg. --size 512x512)");
                return false;
            }
        }

        if (mode == ServerEndpoint::unset) {
            LOG_ERROR("error: no endpoint specified (need to specify --generate or --edit)");
            return false;
        }

        if (mode == ServerEndpoint::edit) {
            if (edit_images.size() < 1) {
                LOG_ERROR("error: edit endpoint requires at least one image (--image FILENAME)");
                return false;
            }
        }
        else {  // !ServerEndpoint::edit
            if (edit_images.size() > 0) {
                LOG_ERROR("error: image requires edit endpoint (--edit)");
                return false;
            }
            if (edit_mask.filename().size() > 0) {
                LOG_ERROR("error: mask requires edit endpoint (--edit)");
                return false;
            }
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDClientParams {\n"
            << "  server_ip: " << server_ip << ",\n"
            << "  server_port: \"" << server_port << "\",\n"
            << "  endpoint: \"";
        if (mode == ServerEndpoint::generate) oss << "/v1/images/generations";
        if (mode == ServerEndpoint::edit) oss << "/v1/images/edits";
        oss << "\",\n"
            << "}";
        return oss.str();
    }
};

void print_usage(char const* progname, const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << progname << " [options]\n\n";
    std::cout << "Client Options:\n";
    options_list[0].print();
//    std::cout << "\nContext Options:\n";
//    options_list[1].print();  // n/a (server side options)
    std::cout << "\nDefault Generation Options:\n";
    options_list[1].print();
}


json parse_args(int argc, const char** argv, SDClientParams& client_params, SDGenerationParams& gen_params) {
    std::vector<ArgOptions> options_vec = {client_params.get_options(), gen_params.get_options()};
    bool retval = parse_options(argc, argv, options_vec);

    if (client_params.help) {
        print_usage(argv[0], options_vec);
        exit(0);
    }

    if (client_params.version) {
        std::cout << version_string() << "\n";
        exit(0);
    }

    if (retval)
        retval = client_params.process_and_check();
    if (retval)
        retval = gen_params.process_and_check(client_params.sdmode, "");  // lora_model_dir is server side context

    if (!retval)
        exit(1);

    if (gen_params.prompt.empty()) {
        LOG_ERROR("error: no prompt specified");
        exit(1);
    }

    // convert gen_params to json to pass to server
    json gen_json = options_to_json(gen_params, options_vec[1]);
    return gen_json;
}


void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDClientParams* client_params = (SDClientParams*)data;
    log_print(level, log, client_params->verbose, client_params->color);
}


size_t next_offset = 0;

// if no output file is specified, try to find the next image not named 'prefix-?.ext'
std::string next_autoincr_file(std::string const& prefix, std::string const& ext) {
    size_t offset = next_offset;
    std::string outfile;

    do {
        outfile = prefix + "-" + std::to_string(offset++) + "." + ext;
    } while (fs::exists(outfile));

    next_offset = offset;  // cache for batch mode

    return outfile;
}


/** get output file name (todo: regex matching like in sd-cli for image sequence (not batch_count)) **/
file_data get_output_file(SDClientParams const& client_params) {
    // determine final output target file (either specified or autoincr with optional leading path)
    file_data outfile;
    {
        std::string fname;
        if ( !client_params.output_path.empty() ) {
            if ( !fs::is_directory(client_params.output_path) ) {
                LOG_ERROR("error: `%s`: output directory not found", client_params.output_path.c_str());
                exit(1);
            }
            if ( client_params.output_file.empty() )
                fname = next_autoincr_file(client_params.output_path + "/output", client_params.format);
            else
                fname = client_params.output_path + "/" + client_params.output_file + "." + client_params.format;
        }
        else {  // no output path specified
            if ( client_params.output_file.empty() )
                fname = next_autoincr_file("output", client_params.format);
            else
                fname = client_params.output_file + "." + client_params.format;
        }
        if (!outfile.prepare_save(fname, client_params.valid_file_types, client_params.overwrite))
            exit(1);  // fatal error if we can't output to the file
    }
    return outfile;
}


std::atomic<bool> client_cancel = false;  // used to signal client shutdown gracefully


int main(int argc, const char** argv) {
    if (argc < 1)
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }

    SDClientParams client_params;
    SDGenerationParams gen_params;
    json gen_json = parse_args(argc, argv, client_params, gen_params);

    sd_set_log_callback(sd_log_cb, (void*)&client_params);
    log_verbose = client_params.verbose;
    log_color   = client_params.color;

    LOG_DEBUG("version: %s", version_string().c_str());
    LOG_DEBUG("%s", sd_get_system_info());
    LOG_DEBUG("%s", client_params.to_string().c_str());

    std::vector<file_data> outfile_list;
    outfile_list.resize(gen_params.batch_count);
    for (size_t pos = 0; pos < gen_params.batch_count; ++pos)
        outfile_list[pos] = get_output_file(client_params);


    std::signal(SIGINT, [](int signal) {
      client_cancel.store(true, std::memory_order_release);
    });

    httplib::Client client(client_params.server_ip, client_params.server_port);


    std::future<httplib::Result> ft = std::async(std::launch::async, [&]() {
        client_cancel.store(false, std::memory_order_release);

        if (client_params.mode == ServerEndpoint::generate) {
            json req;
            req["size"] = client_params.size;
            req["output_format"] = client_params.format;
            req["output_compression"] = client_params.compression;

            //req.update(gen_json);  // merge in gen_json
            req["gen"] = gen_json;  // requires changes to the server

            return client.Post("/v1/images/generations", req.dump(), "text/json");
        }

        if (client_params.mode == ServerEndpoint::edit) {
            // while this is neat, it would be simpler if it operated like generate (b64 encodes, json, etc)
            httplib::UploadFormDataItems req = {
              //{"prompt", client_params.prompt, "", "text/plain"},
                {"gen", gen_json.dump(), "", "text/json"},
                {"size", client_params.size, "", "text/plain"},
                {"output_format", client_params.format, "", "text/plain"},
                {"output_compression", std::to_string(client_params.compression), "", "text/plain"}
            };

            for ( auto const& data : client_params.edit_images ) {
                std::unique_ptr<std::vector<uint8_t>> bytes = data.load();

                if (!bytes)
                    return httplib::Result();

                req.push_back({"image[]", std::string(bytes->begin(), bytes->end()), data.filename(), std::string("image/")+data.extension()});
            }

            if ( client_params.edit_mask.filename().size() > 0 ) {
                auto const& data = client_params.edit_mask;
                std::unique_ptr<std::vector<uint8_t>> bytes = data.load();

                if (!bytes)
                    return httplib::Result();

                req.push_back({"image[]", std::string(bytes->begin(), bytes->end()), data.filename(), std::string("image/")+data.extension()});
            }
            return client.Post("/v1/images/edits", req);
        }
        return httplib::Result();
    });

    std::future_status ft_status;
    do {
        if (!ft.valid()) break;
        ft_status = ft.wait_for(std::chrono::milliseconds(1000));
    } while (!client_cancel.load(std::memory_order_relaxed) && ft_status != std::future_status::ready);

    if (!ft.valid()) {
        LOG_ERROR("server error: unexpected error");
        return 1;
    }

    if (ft_status != std::future_status::ready) {
      // cancel triggered, cleanup
      client.stop();
      ft.wait();
      LOG_INFO("operation cancelled, shutting down...");
      return 1;
    }
    auto res = ft.get();
    client.stop();

    if ( !res ) {
        LOG_ERROR("server error: %s", httplib::to_string(res.error()).c_str());
        return 1;
    }

    std::string content_type = res->get_header_value("Content-Type");
    if (content_type != "application/json") {
        LOG_ERROR("error: unexpected content type: %s", content_type);
        return 1;
    }

    json body = json::parse(res->body);
    if (body.contains("error")) {  // report custom server error message
        LOG_ERROR("server error: %s: %s", httplib::status_message(res->status), body["error"].get<std::string>().c_str());
        return 1;
    }

    if (res->status != 200) {  // status != success, but no json error in body
        LOG_ERROR("server error: %s", httplib::status_message(res->status));
        return 1;
    }

    if (body["data"].size() != gen_params.batch_count) {
        LOG_ERROR("error: expected %i images, got %i instead", gen_params.batch_count, body["data"].size());
        return 1;
    }

    if (!body.contains("data")) {
        LOG_ERROR("error: server appears to have sent no data");
        return 1;
    }

    size_t pos = 0;  // only support one image write for now (needs updating)

    for (size_t pos = 0; pos < body["data"].size(); ++pos) {
        if (!body["data"][pos].contains("b64_json")) {
            LOG_ERROR("error: image %i contains no data; skipping", pos);
            continue;
        }
        std::string b64 = body["data"][pos]["b64_json"];
        std::vector<uint8_t> image_bytes = base64_decode(b64);
        LOG_INFO("writing output to: `%s`...", outfile_list[pos].filename().c_str());
        outfile_list[pos].save(image_bytes);
    }

    return 0;
}

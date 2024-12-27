#include "config.h"

#if defined(HAVE_FILESYSTEM)
#include <filesystem>
namespace fs = std::filesystem;
#elif defined(HAVE_EXPERIMENTAL_FILESYSTEM)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

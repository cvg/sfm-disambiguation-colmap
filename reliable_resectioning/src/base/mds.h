#ifndef COLMAP_SRC_BASE_MDS_H_
#define COLMAP_SRC_BASE_MDS_H_

#include <set>

#include "base/database_cache.h"
#include "base/image.h"
#include "util/alignment.h"
#include "util/misc.h"
#include <Eigen/Core>
#include <float.h>
#include <jsoncpp/json/json.h>
#include <math.h>

namespace colmap {

class MDS {
 public:
  MDS(const std::string database_path, const std::string metric,
      const DatabaseCache* database_cache);
  float GetMinImageDistanceScore(const image_t image_id,
                                 const Image& image) const;
  void SetRegisteredImages(const std::vector<image_t>& reg_image_ids);
  int RegisteredImageRank(const Image& im1, const Image& im2) const;
  int AdaptiveK(const Image& im1) const;

 private:
  void CalculateDistances();
  size_t min_distance_;
  std::string database_path_;
  std::string metric_;
  const DatabaseCache* database_cache_;
  static std::set<std::string> registered_images_;
};

struct Coordinate {
  double x, y;
  std::string image_name;

  Coordinate(std::string name, double paramx, double paramy)
      : image_name(name), x(paramx), y(paramy) {}
};
}  // namespace colmap

#endif  // COLMAP_SRC_BASE_VISIBILITY_PYRAMID_H_

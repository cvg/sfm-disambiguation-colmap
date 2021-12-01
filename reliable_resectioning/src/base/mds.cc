#include "base/mds.h"
#include "base/correspondence_graph.h"
#include "util/logging.h"
#include "util/math.h"

#include <fstream>
#include <iostream>
#include <math.h>

namespace colmap {

std::set<std::string> MDS::registered_images_;
std::map<std::string, std::map<std::string, float>> aam_distances_map;
std::string database_path_;
std::vector<Image> images;
std::vector<std::string> image_names;

MDS::MDS(const std::string database_path, std::string metric,
         const DatabaseCache *database_cache) {
  std::cout
      << "*********************************************************************"
         "**************************************************************"
      << std::endl;
  std::cout << "******************************************************** "
               "Initializing MDS "
               "*********************************************************"
            << std::endl;
  std::cout << "******************************************************** "
            << metric
            << " *********************************************************"
            << std::endl;
  std::cout
      << "*********************************************************************"
         "**************************************************************"
      << std::endl;
  database_path_ = database_path;
  metric_ = metric;
  database_cache_ = database_cache;

  Database database(database_path_ + "/database.db");
  images = database.ReadAllImages();

  for (std::vector<Image>::iterator it = images.begin(); it != images.end();
       ++it) {
    image_names.push_back(it->Name());
  }

  std::sort(image_names.begin(), image_names.end());

  CorrespondenceGraph sg = database_cache->CorrespondenceGraph();
  for (auto const &im1 : image_names) {
    std::map<std::string, float> im1_aam_map;
    for (auto const &im2 : image_names) {
      if (im1.compare(im2) == 0) {
        im1_aam_map.insert(std::make_pair(im2, 0.0));
      } else if (im1.compare(im2) > 0) {
        im1_aam_map.insert(std::make_pair(
            im2, aam_distances_map.find(im2)->second.find(im1)->second));
      } else {
        Image im1_ = database.ReadImageWithName(im1);
        Image im2_ = database.ReadImageWithName(im2);
        if (!sg.ExistsImage(im1_.ImageId()) ||
            !sg.ExistsImage(im2_.ImageId())) {
          continue;
        }
        colmap::FeatureMatches correspondences =
            sg.FindCorrespondencesBetweenImages(im1_.ImageId(), im2_.ImageId());

        colmap::FeatureMatches::iterator correspondence_itr;
        double aam_matches = 0.0;
        for (correspondence_itr = correspondences.begin();
             correspondence_itr != correspondences.end();
             ++correspondence_itr) {
          std::vector<CorrespondenceGraph::Correspondence>
              point_correspondences = sg.FindCorrespondences(
                  im1_.ImageId(), (*correspondence_itr).point2D_idx1);
          if (!sg.HasCorrespondences(im1_.ImageId(),
                                     (*correspondence_itr).point2D_idx1)) {
            continue;
          }
          if (!sg.HasCorrespondences(im2_.ImageId(),
                                     (*correspondence_itr).point2D_idx2)) {
            continue;
          }
          // tuned parameters
          // aam_matches += pow(0.5, point_correspondences.size() - 1);
          aam_matches += pow(0.3, point_correspondences.size() - 1);
        }
        im1_aam_map.insert(std::make_pair(im2, 1.0 / (aam_matches + 0.000001)));
      }
    }
    aam_distances_map.insert(std::make_pair(im1, im1_aam_map));
  }
  std::cout
      << "******************************************************** Finished "
         "Initializing MDS ************************************************"
      << std::endl;
}

void MDS::CalculateDistances() {}

void MDS::SetRegisteredImages(const std::vector<image_t> &reg_image_ids) {
  registered_images_.clear();
  for (int i = 0; i < reg_image_ids.size(); i++) {
    unsigned int image_id = reg_image_ids[i];
    const Image &image = database_cache_->Image(image_id);
    registered_images_.insert(image.Name().c_str());
  }
}

float MDS::GetMinImageDistanceScore(const image_t image_id,
                                    const Image &image) const {
  float min_aam_distance = 100000.0;
  std::string im1 = image.Name().c_str();
  std::string min_aam_distance_image;
  std::map<std::string, float> im1_aam_map =
      aam_distances_map.find(im1)->second;

  for (int j = 0; j < image_names.size(); j++) {
    std::string im2 = image_names[j];
    float im1_im2_aam_distance = im1_aam_map.find(im2)->second;

    if ((im1_im2_aam_distance < min_aam_distance) && (im1.compare(im2) != 0) &&
        (std::find(registered_images_.begin(), registered_images_.end(), im2) !=
         registered_images_.end())) {
      min_aam_distance = im1_im2_aam_distance;
      min_aam_distance_image = im2;
    }
  }
  return 1.0 / min_aam_distance;
}

int MDS::RegisteredImageRank(const Image &im1, const Image &im2) const {
  int current_aam_rank = 1;
  std::map<std::string, float> im1_aam_map =
      aam_distances_map[im1.Name().c_str()];
  float im2_aam_distance = im1_aam_map[im2.Name().c_str()];

  std::set<std::string>::iterator itr;
  for (itr = registered_images_.begin(); itr != registered_images_.end();
       ++itr) {
    if (im1_aam_map[*itr] < im2_aam_distance) {
      current_aam_rank += 1;
    }
  }
  return current_aam_rank;
}

int MDS::AdaptiveK(const Image &im1) const {
  int k_aam = 0;
  // tuned parameters
  // float tau_value = 2.0;
  float tau_value = 1.3;
  std::cout << "\t\tAdaptiveK Parameter: " << tau_value << std::endl;
  std::map<std::string, float> im1_aam_map =
      aam_distances_map[im1.Name().c_str()];

  float closest_registered_image_aam_distance = FLT_MAX;
  std::string closest_registered_aam_image = im1.Name().c_str();

  // Get closest registered image distance
  std::set<std::string>::iterator itr;
  for (itr = registered_images_.begin(); itr != registered_images_.end();
       ++itr) {
    if (im1_aam_map[*itr] < closest_registered_image_aam_distance) {
      closest_registered_image_aam_distance = im1_aam_map[*itr];
      closest_registered_aam_image = *itr;
    }
  }

  for (itr = registered_images_.begin(); itr != registered_images_.end();
       ++itr) {
    if (im1_aam_map[*itr] <=
        tau_value * closest_registered_image_aam_distance) {
      k_aam += 1;
    }
  }
  return k_aam;
}
} // namespace colmap

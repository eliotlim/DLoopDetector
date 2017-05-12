/**
 * File: demo_brief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <string>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines BriefVocabulary
#include "DLoopDetector.h" // defines BriefLoopDetector
#include <DVision/DVision.h> // Brief

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "demoDetector.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;

/// FORB Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
FORBVocabulary;

/// FORB Database
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB>
FORBDatabase;

// ----------------------------------------------------------------------------

static const char *VOC_FILE = "./resources/ORBvoc.txt";
static const char *IMAGE_DIR = "./resources/images";
static const char *POSE_FILE = "./resources/pose.txt";
static const int IMAGE_W = 640; // image size
static const int IMAGE_H = 480;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// This functor extracts BRIEF descriptors in the required format
class FORBExtractor: public FeatureExtractor<FORB::TDescriptor>
{
public:
  /**
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys, std::vector<cv::Mat> &descriptors) const;

  /**
   * Creates the FORB extractor
   * @param pattern_file
   */
  FORBExtractor();

  /**
   * Transforms descriptors from cv::Mat of dimensions (1000, 32) to std::vector<cv::Mat> of dimensions (1, 32)
   * @param descriptors descriptor matrix
   * @param output_descriptors descriptor vector
   */
  void transformDescriptors (cv::Mat &descriptors, std::vector<cv::Mat> &output_descriptors) const;

private:

  /// FORB descriptor extractor
  cv::Ptr<cv::ORB> m_orb_;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int main()
{
  // prepares the demo
  demoDetector<FORBVocabulary, FORBLoopDetector, FORB::TDescriptor>
    demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);

  try
  {
    // run the demo with the given functor to extract features
    FORBExtractor extractor = FORBExtractor();
    demo.run("FORB", extractor);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}

// ----------------------------------------------------------------------------

FORBExtractor::FORBExtractor()
{
    m_orb_ = cv::ORB::create (1000);
}

// ----------------------------------------------------------------------------

void FORBExtractor::operator() (const cv::Mat &im,
  vector<cv::KeyPoint> &keys, vector<cv::Mat> &descriptors) const
{
  // extract FAST keypoints with opencv
  const int fast_th = 20; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);

  // compute their ORB descriptor
  cv::Mat descriptors_temp;
  m_orb_->compute(im, keys, descriptors_temp);

  // transform ORB descriptor matrix into vector
  transformDescriptors (descriptors_temp, descriptors);

}

// ----------------------------------------------------------------------------

void FORBExtractor::transformDescriptors (cv::Mat& mat, std::vector<cv::Mat>& output_descriptors) const {
    // CV_8UC1 matrix only
    if (mat.type() == 0) {
        output_descriptors.clear();
        for (int row = 0; row < mat.rows; row++) {
            cv::Rect regionOfInterest (0, row, mat.cols, 1);
            output_descriptors.push_back (cv::Mat (mat, regionOfInterest));
        }
    }
}

// ----------------------------------------------------------------------------

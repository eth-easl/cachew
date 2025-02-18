//
// Created by damien-aymon on 26.05.21.
//

#include "cache_model.h"

namespace tensorflow {
namespace data {
namespace cache_model {

//#define GCS_BYTES_PER_SECOND 153639507.7
#define GCS_BYTES_PER_SECOND 153639507.7
//#define GCS_BYTES_PER_SECOND 277813518.1 // From reading imagenet data.

#define GLUSTER_BYTES_PER_MS 942386.7

#define CACHE_MODEL_TABLE_SIZE 11

static uint64  cache_table_row_sizes[CACHE_MODEL_TABLE_SIZE] =
    { 5000000, 10000000, 15000000, 20000000,
      25000000, 30000000,                                    // 10-08-21
      50000000, 75000000, 100000000, 200000000, 400000000 }; // 5-200MB

static double cache_table_row_times[CACHE_MODEL_TABLE_SIZE] = {
    5.626863, 11.9417165, 13.677828250000001,
    22.252481250000002, 26.33922275, 30.49620075, // 10-08-21
    75.09628599999999, 112.799945, 155.009453, 306.62446275, 624.65508275 };

//152.70708925, 306.62446275, 624.65508275

/*
static uint64 cache_table_row_sizes[CACHE_MODEL_TABLE_SIZE] =
    {4, 64, 512, 1024, 4096, 8192, 10000, 50000, 100000, 200000, 400000, 800000,
     1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000, // 1-5MB
     6000000, 8000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000, 40000000, 45000000, 50000000,
     55000000, 60000000, // 6-60MB
    };

static double cache_table_row_times[CACHE_MODEL_TABLE_SIZE] = {
    4.66038279400527E-05,
    4.68934947200614E-05,
    4.92096787100309E-05,
    4.72388496799249E-05,
    4.89650223699573E-05,
    5.29008085597889E-05,
    5.42046370398748E-05,
    7.78661772998748E-05,
    0.0001021462888,
    8.7556203480126E-05,
    0.00022149618031,
    0.0006084230441,
    0.00080073113117,
    0.00099394794622,
    0.00132328174699,
    0.00184654361403,
    0.00198722723068,
    0.00246736134616,
    0.00264586374527,
    0.00298529173191,
    0.00307663200038,
    0.003311179368301,
    0.004510265794701,
    0.005721448572399,
    0.008696249261,
    0.0109545122866,
    0.013879507301199,
    0.016812270911098,
    0.0214489650292,
    0.0245194503315,
    0.0277513987415,
    0.030901952619399,
    0.034936749091699,
    0.037763399385101,
};*/

/**
 * Performs a linear interpolation between two points.
 * Returns the y-value for the provided new_x
 * @param min_x
 * @param max_x
 * @param min_y
 * @param max_y
 * @param new_x
 * @return
 */
double lin_interpolate(
    uint64 min_x, uint64 max_x, double min_y, double max_y, uint64 new_x) {
  double delta_x = max_x - min_x;
  assert(delta_x > 0);
  double delta_y = max_y - min_y;
  double slope = delta_y / delta_x;
  double y_offset = min_y - (min_x * slope);

  return new_x * slope + y_offset;
}

/**
 * Returns the expected time to read a row from cache given the row size,
 * in seconds.
 * @param row_size
 * @return
 */
double GetTimePerRow(uint64 row_size) {
  // 1 - Start with special cases
  // 1a - row size smaller than smallest available in table
  if (row_size < cache_table_row_sizes[0]) {
    VLOG(0) << "EASL caching model:"
               "outside of cache table range with row size " << row_size <<
               " using 900MiB/s throughput as approximate";

    return row_size / GLUSTER_BYTES_PER_MS;
  }

  // 1b - row size larger than smallest available in table
  if (row_size > cache_table_row_sizes[CACHE_MODEL_TABLE_SIZE - 1]) {
    VLOG(0) << "EASL caching model:"
               "outside of cache table range with row size " << row_size;

    uint64 min_x = cache_table_row_sizes[CACHE_MODEL_TABLE_SIZE - 2];
    uint64 max_x = cache_table_row_sizes[CACHE_MODEL_TABLE_SIZE - 1];
    double min_y = cache_table_row_times[CACHE_MODEL_TABLE_SIZE - 2];
    double max_y = cache_table_row_times[CACHE_MODEL_TABLE_SIZE - 1];

    return lin_interpolate(min_x, max_x, min_y, max_y, row_size); // sec to milisec...
  }

  // 2 - Linearly walk the table to find the proper index
  int index = 1;
  while (index < CACHE_MODEL_TABLE_SIZE) {
    if (row_size < cache_table_row_sizes[index]) {
      uint64 min_x = cache_table_row_sizes[index - 1];
      uint64 max_x = cache_table_row_sizes[index];
      double min_y = cache_table_row_times[index - 1];
      double max_y = cache_table_row_times[index];

      return lin_interpolate(min_x, max_x, min_y, max_y, row_size); // sec to milisec...
    }
    index++;
  }

  VLOG(0) << "Should not end up here...";
  DCHECK(false);
};


double GetGCSThrouhgput(double alpha) {
  return GCS_BYTES_PER_SECOND * alpha;
}

} // cache_model
} // data
} // tensorflow

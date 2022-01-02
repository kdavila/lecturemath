//Compile using:
//	gcc -m64 -shared accessmath_lib.c -o accessmath_lib.so
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int speaker_detection_handle_frame(unsigned char *frame, unsigned char *last_frame, 
									int width, int height, int channels, int threshold, int jump_cells, 
									double *change_boundaries, double *change_avg, double* change_deviation){


	change_avg[0] = 0.0;
	change_avg[1] = 0.0;
	int total_changes = 0;
	int changed;
	int row, col, color, offset, i;
	
	int min_changed_x = width + 1;
    int max_changed_x = -1;
    int min_changed_y = height + 1;
	int max_changed_y = -1;
	
	double count_values_x[width];
	double count_values_y[height];
	
	for (i = 0; i < width; i++){
		count_values_x[i] = 0.0;
	}
	for (i = 0; i < height; i++){
		count_values_y[i] = 0.0;
	}
	
		
	//For every pixel in the image...
	for (row = 0; row < height; row += jump_cells){
		for (col = 0; col < width; col += jump_cells){
			//Start of the pixel...
			offset = ((row * width) + col) * channels;
			
			//Check if at least one colour channel changed more than threshold...
			changed = 0;
			for (color = 0; color < channels; color++){
				if (abs((int)last_frame[offset + color] - (int)frame[offset + color]) > threshold){
					//Change found...
					changed = 1;
					break;
				}	
			}
			
			if (changed > 0){
				//changed ... motion detected...
                if (col < min_changed_x){
                    min_changed_x = col;
				}
                if (col > max_changed_x){
                    max_changed_x = col;
				}
                if (row < min_changed_y){
                    min_changed_y = row;
				}
                if (row > max_changed_y){
                    max_changed_y = row;
				}

                change_avg[0] += col; //x
                change_avg[1] += row; //y
                
                total_changes ++;
				
				//Add to histogram....
				count_values_x[col]++;
				count_values_y[row]++;
			}
		}
	}
	
	//Copy boundaries...
	change_boundaries[0] = min_changed_x;
	change_boundaries[1] = max_changed_x;
	change_boundaries[2] = min_changed_y;
	change_boundaries[3] = max_changed_y;
	
	//Now get average and Standard Deviation of changes...
	if ( total_changes > 0 ){
		change_avg[0] /= (double)total_changes; //x
		change_avg[1] /= (double)total_changes; //y
		
		//now get the variance of changed points (using histograms)...		
		change_deviation[0] = 0.0;
		change_deviation[1] = 0.0;
		//...cols...
		for (col = 0; col < width; col++){
			change_deviation[0] += ((double)col - change_avg[0]) * ((double)col - change_avg[0]) * count_values_x[col];
		}
		//...rows...
		for (row = 0; row < height; row++){
			change_deviation[1] += ((double)row - change_avg[1]) * ((double)row - change_avg[1]) * count_values_y[row];
		}
		
		change_deviation[0] /= (double)total_changes; //x
		change_deviation[1] /= (double)total_changes; //y
		
		change_deviation[0] = sqrt(change_deviation[0]);
		change_deviation[1] = sqrt(change_deviation[1]);	
	} else {
		change_deviation[0] = 0.0;
		change_deviation[1] = 0.0;
	}
	
	return total_changes;
}

void regionCumulativeDistribution(unsigned char* grayscale, int width, int height, int min_x, int max_x,
                                  int min_y, int max_y, double slope_max, double* output){
    int hist[256];
    int i;

    //First, compute the histogram
    for (i = 0; i < 256; i++){
        hist[i] = 0;
    }

    int x;
    int y;
    int offset;
    for (y = min_y; y <= max_y; y++)
    {
        offset = width * y + min_x;

        for (x = min_x; x <= max_x; x++)
        {
            hist[grayscale[offset]]++;

            offset++;
        }
    }

    // Compute the raw cumulative distribution
    int count = 0;
    for (i = 0; i < 256; i++)
    {
        count += hist[i];
        output[i] = count;
    }


    // Normalize cumulative distribution
    for (i = 0; i < 256; i++)
    {
        output[i] /= count;
    }

    // Apply contrast limit ...
    if (slope_max > 0.0){
        double dh = 0.0;
        double diff;

        for (i = 0; i < 255; i++)
        {
            diff = output[i + 1] - output[i] - dh - slope_max;
            dh += (diff < 0.0 ? 0.0 : diff);
            output[i + 1] -= dh;

        }

        //Center ....
        double add = (1.0 - (output[255] - output[0])) / 2.0;
        for (i = 0; i < 256; i++)
        {
            output[i] += add;
        }
    }
}

int adapthisteq(unsigned char* grayscale, int width, int height, double slope,
                int grid_x, int grid_y, unsigned char* output){

    int min_size_y = height / grid_y;
    int min_size_x = width / grid_x;

    //Compute the contrast-limited, centered, cumulative distribution on each cell of the grid
    double* dist = (double *)malloc(grid_x * grid_y * 256 * sizeof(double));
    int x_limits_min[grid_x];
    int x_limits_max[grid_x];
    int x_limits_mid[grid_x];
    int y_limits_min[grid_y];
    int y_limits_max[grid_y];
    int y_limits_mid[grid_y];

    int rx;
    int ry;
    int start_x = 0;
    int start_y = 0;
    int end_x;
    int end_y;
    int mod_x = width % grid_x;
    int mod_y = height % grid_y;
    for (rx = 0; rx < grid_x; rx++){
        end_x = start_x + min_size_x + (rx < mod_x ? 1 : 0) - 1;

        x_limits_min[rx] = start_x;
        x_limits_max[rx] = end_x;
        x_limits_mid[rx] = (int)round((start_x + end_x) / 2.0);

        start_y = 0;
        for (ry = 0; ry < grid_y; ry++){
            end_y = start_y + min_size_y + (ry < mod_y ? 1 : 0) - 1;

            y_limits_min[ry] = start_y;
            y_limits_max[ry] = end_y;
            y_limits_mid[ry] = (int)round((start_y + end_y) / 2.0);

            regionCumulativeDistribution(grayscale, width, height, start_x, end_x,
                                         start_y, end_y, slope, dist + (ry * grid_x + rx) * 256);

            start_y = end_y + 1;
        }

        start_x = end_x + 1;
    }

    //Now, interpolate the distributions for each pixel of the output...
    int x;
    int y;
    int current_x = 0;
    int current_y = 0;
    double* local_dist;
    double* d00;
    double* d01;
    double* d10;
    double* d11;
    unsigned char tone;
    unsigned char eq_tone;
    int y0;
    int y1;
    int x0;
    int x1;
    double wy1;
    double wx1;
    for (x = 0; x < width; x++){
        if (x > x_limits_max[current_x]){
            current_x ++;
        }

        current_y = 0;
        for (y = 0; y < height; y++){
            if (y > y_limits_max[current_y]){
                current_y ++;
            }

            local_dist = dist + (grid_x * current_y + current_x) * 256;
            tone = *(grayscale + (width * y + x));
            eq_tone = 0;

            //check current case ...
            if ((current_x == 0 && x <= x_limits_mid[current_x]) ||
                (current_x == grid_x - 1 &&  x >= x_limits_mid[current_x])){
                //on first tile, until middle pixel or last tile at or after middle pixel

                if ((current_y == 0 && y <= y_limits_mid[current_y]) ||
                    (current_y == grid_y - 1 && y >= y_limits_mid[current_y])){

                    //Any corner... use just one patch
                    eq_tone = (unsigned char)round(local_dist[tone] * 255);
                } else {
                    //somewhere between the top and bottom... interpolate vertically
                    y0 = current_y - (y <= y_limits_mid[current_y] ? 1 : 0);
                    y1 = y0 + 1;

                    wy1 = (y - y_limits_mid[y0]) / (double)(y_limits_mid[y1] - y_limits_mid[y0]);

                    d00 = dist + (grid_x * y0 + current_x) * 256;
                    d01 = dist + (grid_x * y1 + current_x) * 256;

                    //Linear interpolation
                    eq_tone = (unsigned char)round((d00[tone] * (1.0 - wy1) + d01[tone] * wy1) * 255);
                }
            } else if ((current_y == 0 && y <= y_limits_mid[current_y]) ||
                       (current_y == grid_y - 1 && y >= y_limits_mid[current_y])) {
                //Top or bottom ... interpolate horizontally
                x0 = current_x - (x <= x_limits_mid[current_x] ? 1 : 0);
                x1 = x0 + 1;

                wx1 = (x - x_limits_mid[x0]) / (double)(x_limits_mid[x1] - x_limits_mid[x0]);

                d00 = dist + (grid_x * current_y + x0) * 256;
                d10 = dist + (grid_x * current_y + x1) * 256;

                //Linear interpolation
                eq_tone = (unsigned char)round((d00[tone] * (1.0 - wx1) + d10[tone] * wx1) * 255);
            } else {
                //Full interpolation
                x0 = current_x - (x <= x_limits_mid[current_x] ? 1 : 0);
                x1 = x0 + 1;
                wx1 = (x - x_limits_mid[x0]) / (double)(x_limits_mid[x1] - x_limits_mid[x0]);

                y0 = current_y - (y <= y_limits_mid[current_y] ? 1 : 0);
                y1 = y0 + 1;
                wy1 = (y - y_limits_mid[y0]) / (double)(y_limits_mid[y1] - y_limits_mid[y0]);

                d00 = dist + (grid_x * y0 + x0) * 256;
                d01 = dist + (grid_x * y1 + x0) * 256;
                d10 = dist + (grid_x * y0 + x1) * 256;
                d11 = dist + (grid_x * y1 + x1) * 256;

                eq_tone = (unsigned char)round((d00[tone] * (1.0 - wx1) * (1.0 - wy1) +
                                                d01[tone] * (1.0 - wx1) * wy1 +
                                                d10[tone] * wx1 * (1.0 - wy1) +
                                                d11[tone] * wx1 * wy1) * 255);
            }

            //Un-comment to see the tiles ...
            //eq_tone =  (unsigned char)(local_dist[tone] * 255);

            *(output + (width * y + x)) = eq_tone;

        }
    }

    // Free memory ...
    local_dist = 0;
    d00 = 0;
    d01 = 0;
    d10 = 0;
    d11 = 0;
    free(dist);

    return 0;
}

int combine_results(unsigned char* only_board, unsigned char* equalized, int width, int height,
                    unsigned char threshold, unsigned char* final_content){
    int r;
    int c;
    unsigned char* in_equalized = equalized;
    unsigned char* in_only_board = only_board;
    unsigned char* output = final_content;

    for (r = 0; r < height; r++){
        for (c = 0; c < width; c++){
            if (*in_only_board > 128){
                *output = 0;
            } else {
                *output = (*in_equalized < threshold ? 255 : 0);
            }

            output++;
            in_equalized++;
            in_only_board++;
        }
    }

    return 0;
}


int CC_AgeBoundaries(int* labels, float* ages, int width, int height, int count_labels,
                      int* out_mins_y, int* out_maxs_y, int* out_mins_x, int* out_maxs_x,
                      int* out_counts, float* output_age)
{

    //Initialize output ...
    int i;
    for (i = 0;  i < count_labels; i++)
    {
        out_mins_y[i] = height;
        out_maxs_y[i] = 0;

        out_mins_x[i] = width;
        out_maxs_x[i] = 0;

        out_counts[i] = 0;
        output_age[i] = -1.0f; // Invalid value ...
    }

    int x, y;
    int idx = 0;
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            if (labels[idx] > 0){
                int cc_id = labels[idx] - 1;

                if (out_mins_y[cc_id] > y){
                    out_mins_y[cc_id]  = y;
                }

                if (out_maxs_y[cc_id] < y){
                    out_maxs_y[cc_id] = y;
                }

                if (out_mins_x[cc_id] > x){
                    out_mins_x[cc_id]  = x;
                }

                if (out_maxs_x[cc_id] < x){
                    out_maxs_x[cc_id] = x;
                }

                out_counts[cc_id]++;

                if (output_age[cc_id] < 0.0f || ages[idx] < output_age[cc_id]){
                    output_age[cc_id] = ages[idx];
                }
            }
            idx ++;
        }
    }


    return 0;
}
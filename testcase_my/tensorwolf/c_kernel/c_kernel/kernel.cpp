#define DLLEXPORT extern "C" __declspec(dllexport)
#include <iostream>

// Rho Aias!!!
	
DLLEXPORT 
int correlate2d(
		float* z,
		int batchs,
		int z_h,
		int z_w,
		int i_c,
		int h_step,
		int w_step,
		float* f,
		int f_h,
		int f_w,
		int o_c,
		float* o,
		int o_h,
		int o_w
	) {
	/*
	enumarate dims
	b,oh,ow,ih,iw,ic,oc,
	*/
	int o_b_size = o_h*o_w*o_c;
	int o_oh_size = o_w*o_c;
	int f_fh_size = f_w*i_c*o_c;
	int f_fw_size = i_c*o_c;
	int z_b_size = z_h*z_w*i_c;
	int z_oh_size = h_step*z_w*i_c;
	int z_ow_size = w_step*i_c;
	int z_fh_size = z_w*i_c;

	//std::cout << "Receive" << std::endl;
	for (int b = 0; b < batchs; ++b) {
		float* o_b = o + b*o_b_size;
		float* z_b = z + b*z_b_size;

		for (int oh = 0; oh < o_h; ++oh) {
			float* o_oh = o_b + oh*o_oh_size;
			float* z_oh = z_b + oh*z_oh_size;

			for (int ow = 0; ow < o_w; ++ow) {
				float* o_ow = o_oh + ow*o_c;
				float* z_ow = z_oh + ow*z_ow_size;

				for (int fh = 0; fh < f_h; ++fh) {
					float* f_fh = f + fh * f_fh_size;
					float* z_fh = z_ow + fh * z_fh_size;

					for (int fw = 0; fw < f_w; ++fw) {
						float* f_fw = f_fh + fw * f_fw_size;
						float* z_fw = z_fh + fw * i_c;

						for (int ic = 0; ic < i_c; ++ic) {
							float* f_ic = f_fw + ic * o_c;

							for (int oc = 0; oc < o_c; ++oc) {
								// o[b][oh][ow][oc] += z[b][ih][iw][ic] * f[fh][fw][ic][oc]
								o_ow[oc] += z_fw[ic] * f_ic[oc];
								// std::cout << (o_ow - o) + oc << '=' << (z_fw - z) + ic << '+' << (f_ic - f) + oc << std::endl;
							}
						}
					}
				}
			}
		}
	}
	//std::cout << "Receive" << std::endl;
	return 0;
}

DLLEXPORT
int conv2d_filter_gradient(
		float* z,
		int batchs,
		int z_h,
		int z_w,
		int i_c,
		float* f,
		int f_h,
		int f_w,
		int o_c,
		float* o,
		int o_h,
		int o_w
	) {
	/*
		enumarate dims
		b, o_h, o_w, f_h, f_w, i_c, o_c
	*/
	int o_oh_size = o_w*i_c*o_c;
	int o_ow_size = i_c*o_c;
	int z_b_size = z_h*z_w*i_c;
	int z_oh_size = z_w*i_c;
	int z_fh_size = z_w*i_c;
	int f_b_size = f_h*f_w*o_c;
	int f_fh_size = f_w*o_c;
	
	for (int b = 0; b < batchs; ++b) {
		float* z_b = z + b*z_b_size;
		float* f_b = f + b*f_b_size;

		for (int oh = 0; oh < o_h; ++oh) {
			float* o_oh = o + oh*o_oh_size;
			float* z_oh = z_b + oh*z_oh_size;

			for (int ow = 0; ow < o_w; ++ow) {
				float* o_ow = o_oh + ow*o_ow_size;
				float* z_ow = z_oh + ow*i_c;

				for (int fh = 0; fh < f_h; ++fh) {
					float* z_fh = z_ow + fh *z_fh_size;
					float* f_fh = f_b + fh*f_fh_size;

					for (int fw = 0; fw < f_w; ++fw) {
						float* z_fw = z_fh + fw *i_c;
						float* f_fw = f_fh + fw *o_c;

						for (int ic = 0; ic < i_c; ++ic) {
							float* o_ic = o_ow + ic*o_c;

							for (int oc = 0; oc < o_c; ++oc) {
								// o[oh][ow][ic][oc] += z[b][oh+fh][ow+fw][ic] * f[b][fh][fw][oc]
								o_ic[oc] += z_fw[ic] * f_fw[oc];
								// std::cout << (o_ic - o) + oc << '=' << (z_fw - z) + ic << '+' << (f_fw - f) + oc << std::endl;
							}
						}
					}	
				}
			}
		}
	}
	return 0;
}

DLLEXPORT
int max_pool_gradient(
		float* g,
		int batchs,
		int g_h,
		int g_w,
		int i_c,
		float* o,
		int h_step,
		int w_step,
		float* z,
		int z_h,
		int z_w
	) {
	int g_b_size = g_h*g_w*i_c;
	int g_gh_size = g_w*i_c;
	int z_b_size = z_h*z_w*i_c;
	int z_gh_size = h_step*z_w*i_c;
	int z_gw_size = w_step*i_c;
	int z_h_size = z_w*i_c;

	for (int b = 0; b < batchs; ++b) {
		float* g_b = g + b*g_b_size;
		float* z_b = z + b*z_b_size;

		for (int gh = 0; gh < g_h; ++gh) {
			float* g_gh = g_b + gh*g_gh_size;
			float* z_gh = z_b + gh*z_gh_size;

			for (int gw = 0; gw < g_w; ++gw) {
				float* g_gw = g_gh + gw * i_c;
				float* z_gw = z_gh + gw * z_gw_size;

				for (int ic = 0; ic < i_c; ++ic) {
					float* z_ic = z_gw + ic;
					float* max_loc = z_ic;

					for (int h = 0; h < h_step; ++h) {
						float* z_h = z_ic + h*z_h_size;

						for (int w = 0; w < w_step; ++w) {
							float* z_w = z_h + w*i_c;

							if ((*z_w) > (*max_loc)) {
								max_loc = z_w;
							}
						}
					}
					o[max_loc - z] += g_gw[ic];
				}
			}
		}
	}
	return 0;
}
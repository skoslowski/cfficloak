/* Copyright (c) 2016, Andrew Leech <andrew@alelec.net>
* All rights reserved.
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* 
*     http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* 
* The full license is also available in the file LICENSE.apache-2.0.txt
*/

#include <stddef.h>

int myint_succ(int i);
int myint_succ2(int i);
int myint_doubled(int i);
int myint_add(int i, int j);
int myint_add2(int i, int j);
int myint_mult(int i, int j);
int* myintp_null(int i);

float myfloat_succ(float i);
float myfloat_add(float i, float j);
float* myfloatp_null(float i);

int set_ptr_succ(int i, int *j);
int set_ptr_add(int i, int *j);
float set_ptrf(float i, float *j);
float incr_ptrf(float *i);
double complicated(int in, 
                   float *out,
                   int *inout,
                   unsigned long long in2,
                   double *inout2);

int myint_add_array(int j, int *a, int n);

typedef struct {
    int x;
    int y;
} point_t;

point_t* make_point(int x, int y);
void del_point(point_t* p);
int point_x(point_t* p);
int point_y(point_t* p);
point_t* point_setx(point_t* p, int x);
point_t* point_sety(point_t* p, int y);
double point_dist(point_t* p1, point_t* p2);

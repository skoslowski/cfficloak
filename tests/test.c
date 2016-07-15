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

#include "test.h"
#include <math.h>
#include <stdlib.h>


/* MyInt test functions */

int myint_succ(int i) {
    return i+1;
}

int myint_succ2(int i) {
    return i+2;
}

int myint_doubled(int i) {
    return i*2;
}

int myint_add(int i, int j) {
    return i+j;
}

int myint_add2(int i, int j) {
    return i+j+j;
}

int myint_mult(int i, int j) {
    return i*j;
}

int* myintp_null(int i) {
    return NULL;
}


/* MyFloat test functions */

float myfloat_succ(float i) {
    return i+1.0;
}

float myfloat_add(float i, float j) {
    return i+j;
}

float* myfloatp_null(float i) {
    return NULL;
}


/* MyIntOut test functions */

int set_ptr_succ(int i, int *j) {
    *j = i+1;
    return 42;
}

int set_ptr_add(int i, int *j) {
    (*j)++;
    return 23;
}

/* MyFloatOut test functions */

float set_ptrf(float i, float *j) {
    *j = i+1.0;
    return 42.0;
}

float incr_ptrf(float *i) {
    *i++;
    return 42.0;
}


/* MyInOut test functions */

double complicated(int in, 
                   float *out,
                   int *inout,
                   unsigned long long in2,
                   double *inout2)
{
    *out = (float)in+1.0;
    (*inout)++;
    *inout2 = (float)in2 + *inout2;
    return 42.0;
}


/* Array passing test functions */
int myint_add_array(int j, int *a, int n)
{
    int i;
    for (i = 0; i < n; i++)
        a[i] += j;
    return 0;
}

/* Struct tests */
point_t* make_point(int x, int y)
{
    point_t* p;
    p = (point_t*)malloc(sizeof(point_t));
    p->x = x;
    p->y = y;
    return p;
}

void del_point(point_t* p)
{
    free(p);
}

int point_x(point_t* p)
{
    return p->x;
}

int point_y(point_t* p)
{
    return p->y;
}

point_t* point_setx(point_t* p, int x)
{
    p->x = x;
    return p;
}

point_t* point_sety(point_t* p, int y)
{
    p->y = y;
    return p;
}

double point_dist(point_t* p1, point_t* p2)
{
    double d = 1.0;
    d = sqrt(pow(p2->x - p1->x, 2) + pow(p2->y - p1->y, 2));
    return d;
}


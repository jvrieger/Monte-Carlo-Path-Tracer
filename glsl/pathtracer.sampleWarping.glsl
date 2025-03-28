
vec3 squareToDiskConcentric(vec2 xi) {
    // Implement Peter Shirley's warping method that better preserves relative sample distances
    // This code is by Peter Shirley and is in the public domain.
    // It maps points in [0,1)^2 to the unit disk centered at the origin.
    // The algorithm is described in A Low Distortion Map Between Disk and Square
    // by Shirley and Chiu 2(3).
    // The code below is much nicer than the code in the original article and is
    // a slightly modification of code by David Cline. It includes two small
    // improvements suggested by Franz and by Greg Ward in a blog discussion
    // http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
    float theta, r;
    float a = 2.f * xi.x - 1.f;
    float b = 2.f * xi.y - 1.f;
    if (a == 0 && b == 0) { // if a, b = 0
        r = 0.f;
        theta = 0.f;
    } else if ((a * a) > (b * b)) { // if a^2 > b^2
        r = a;
        theta = (PI / 4.f) * (b / a);
    } else {
        r = b;
        theta = (PI / 2.f) - (PI / 4.f) * (a / b);
    }

    return vec3(r * cos(theta),
                     r * sin(theta),
                     0.f);
}

vec3 squareToHemisphereCosine(vec2 xi) {
    // Bias the warped samples toward the pole of the hemisphere and away from the base
    // Malley's Method: project a collection of samples uniformly distributed on a disk to the surface of a hemisphere
    // Use code from squareToDiskConcentric() to get a good disk mapping, only changing the z value to consider a 3d object

    vec3 concentricDisk = squareToDiskConcentric(xi);
    concentricDisk.z = sqrt(1.f - (concentricDisk.x * concentricDisk.x) - (concentricDisk.y * concentricDisk.y));

    return concentricDisk;
}

float squareToHemisphereCosinePDF(vec3 sample) {
    if (sample.z >= 0.f) {
        return sample.z / PI;  // Cosine-weighted PDF = cos(theta) / pi
    } else {
        return 0.f;  // not on the hemisphere
    }
}

vec3 squareToSphereUniform(vec2 sample) {
    // Square to Sphere Formula from 461 Notes "Sampling and MC Intro" slide 22
    // x = cos(2𝛑ξ2)√(1 − z2)
    // y = sin(2𝛑ξ2)√(1 − z2)
    // z = 1 − 2ξ1
    float z = 1.f - 2.f * sample.x;
    return vec3(cos(2.f * PI * sample.y) * sqrt(1.f - (z * z)),
                sin(2.f * PI * sample.y) * sqrt(1.f - (z * z)),
                z);
}

/// Float approximate-equality comparison
bool nearly_equal(float a, float b){
    float epsilon = 0.0001f;
    if (a == b) {
        // Shortcut
        return true;
    }

    float diff = abs(a - b);
    if (a * b == 0) {
        // a or b or both are zero; relative error is not meaningful here
        return diff < (epsilon * epsilon);
    }

    return diff / (abs(a) + abs(b)) < epsilon;
}

float squareToSphereUniformPDF(vec3 sample) {
    // if point is on surface of the unit sphere (x^2 + y^2 + z^2 = 1)
    if (nearly_equal((sample.x * sample.x) + (sample.y * sample.y) + (sample.z * sample.z), 1.f)) {
        return INV_FOUR_PI; // uniform PDF over sphere surface area = 1 / 4 pi r^2
    } else {
        return 0.f; // outside the sphere
    }
}



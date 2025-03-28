#version 330 core

// Uniforms
uniform vec3 u_Eye;
uniform vec3 u_Forward;
uniform vec3 u_Right;
uniform vec3 u_Up;

uniform int u_Iterations;
uniform vec2 u_ScreenDims;

uniform sampler2D u_AccumImg;
uniform sampler2D u_EnvironmentMap;

// Varyings
in vec3 fs_Pos;
in vec2 fs_UV;
out vec4 out_Col;

// Numeric constants
#define PI               3.14159265358979323
#define TWO_PI           6.28318530717958648
#define FOUR_PI          12.5663706143591729
#define INV_PI           0.31830988618379067
#define INV_TWO_PI       0.15915494309
#define INV_FOUR_PI      0.07957747154594767
#define PI_OVER_TWO      1.57079632679489662
#define ONE_THIRD        0.33333333333333333
#define E                2.71828182845904524
#define INFINITY         1000000.0
#define OneMinusEpsilon  0.99999994
#define RayEpsilon       0.000005

// Path tracer recursion limit
#define MAX_DEPTH 10

// Light Types
#define AREA_LIGHT 1
#define POINT_LIGHT 2
#define SPOT_LIGHT 3
#define ENVIRONMENT_LIGHT 4

// Area light shape types
#define RECTANGLE 1
#define SPHERE 2

// Material types
#define DIFFUSE_REFL    1
#define SPEC_REFL       2
#define SPEC_TRANS      3
#define SPEC_GLASS      4
#define MICROFACET_REFL 5
#define PLASTIC         6
#define DIFFUSE_TRANS   7

// Data structures
struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Material {
    vec3  albedo;
    float roughness;
    float eta; // For transmissive materials
    int   type; // Refer to the #defines above

    // Indices into an array of sampler2Ds that
    // refer to a texture map and/or roughness map.
    // -1 if they aren't used.
    int   albedoTex;
    int   normalTex;
    int   roughnessTex;
};

struct Intersection {
    float t;
    vec3  nor;
    vec2  uv;
    vec3  Le; // Emitted light
    int   obj_ID;
    Material material;
};

struct Transform {
    mat4 T;
    mat4 invT;
    mat3 invTransT;
    vec3 scale;
};

struct AreaLight {
    vec3 Le;
    int ID;

    // RECTANGLE, BOX, SPHERE, or DISC
    // They are all assumed to be "unit size"
    // and are altered from that size by their Transform
    int shapeType;
    Transform transform;
};

struct PointLight {
    vec3 Le;
    int ID;
    vec3 pos;
};

struct SpotLight {
    vec3 Le;
    int ID;
    float innerAngle, outerAngle;
    Transform transform;
};

struct Sphere {
    vec3 pos;
    float radius;

    Transform transform;
    int ID;
    Material material;
};

struct Rectangle {
    vec3 pos;
    vec3 nor;
    vec2 halfSideLengths; // Dist from center to horizontal/vertical edge

    Transform transform;
    int ID;
    Material material;
};

struct Box {
    vec3 minCorner;
    vec3 maxCorner;

    Transform transform;
    int ID;
    Material material;
};

struct Mesh {
    int triangle_sampler_index;
    int triangle_storage_side_len;
    int num_tris;

    Transform transform;
    int ID;
    Material material;
};

struct Triangle {
    vec3 pos[3];
    vec3 nor[3];
    vec2 uv[3];
};


// Functions
float AbsDot(vec3 a, vec3 b) {
    return abs(dot(a, b));
}

float CosTheta(vec3 w) { return w.z; }
float Cos2Theta(vec3 w) { return w.z * w.z; }
float AbsCosTheta(vec3 w) { return abs(w.z); }
float Sin2Theta(vec3 w) {
    return max(0.f, 1.f - Cos2Theta(w));
}
float SinTheta(vec3 w) { return sqrt(Sin2Theta(w)); }
float TanTheta(vec3 w) { return SinTheta(w) / CosTheta(w); }

float Tan2Theta(vec3 w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

float CosPhi(vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : clamp(w.x / sinTheta, -1.f, 1.f);
}
float SinPhi(vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : clamp(w.y / sinTheta, -1.f, 1.f);
}
float Cos2Phi(vec3 w) { return CosPhi(w) * CosPhi(w); }
float Sin2Phi(vec3 w) { return SinPhi(w) * SinPhi(w); }

Ray SpawnRay(vec3 pos, vec3 wi) {
    return Ray(pos + wi * 0.0001, wi);
}

mat4 translate(vec3 t) {
    return mat4(1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                t.x, t.y, t.z, 1);
}

float radians(float deg) {
    return deg * PI / 180.f;
}

mat4 rotateX(float rad) {
    return mat4(1,0,0,0,
                0,cos(rad),sin(rad),0,
                0,-sin(rad),cos(rad),0,
                0,0,0,1);
}

mat4 rotateY(float rad) {
    return mat4(cos(rad),0,-sin(rad),0,
                0,1,0,0,
                sin(rad),0,cos(rad),0,
                0,0,0,1);
}


mat4 rotateZ(float rad) {
    return mat4(cos(rad),sin(rad),0,0,
                -sin(rad),cos(rad),0,0,
                0,0,1,0,
                0,0,0,1);
}

mat4 scale(vec3 s) {
    return mat4(s.x,0,0,0,
                0,s.y,0,0,
                0,0,s.z,0,
                0,0,0,1);
}

Transform makeTransform(vec3 t, vec3 euler, vec3 s) {
    mat4 T = translate(t)
             * rotateX(radians(euler.x))
             * rotateY(radians(euler.y))
             * rotateZ(radians(euler.z))
             * scale(s);

    return Transform(T, inverse(T), inverse(transpose(mat3(T))), s);
}

bool Refract(vec3 wi, vec3 n, float eta, out vec3 wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = sqrt(1 - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

vec3 Faceforward(vec3 n, vec3 v) {
    return (dot(n, v) < 0.f) ? -n : n;
}

bool SameHemisphere(vec3 w, vec3 wp) {
    return w.z * wp.z > 0;
}

void coordinateSystem(in vec3 v1, out vec3 v2, out vec3 v3) {
    if (abs(v1.x) > abs(v1.y))
            v2 = vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
        else
            v2 = vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
        v3 = cross(v1, v2);
}

mat3 LocalToWorld(vec3 nor) {
    vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return mat3(tan, bit, nor);
}

mat3 WorldToLocal(vec3 nor) {
    return transpose(LocalToWorld(nor));
}

float DistanceSquared(vec3 p1, vec3 p2) {
    return dot(p1 - p2, p1 - p2);
}



// from ShaderToy https://www.shadertoy.com/view/4tXyWN
uvec2 seed;
float rng() {
    seed += uvec2(1);
    uvec2 q = 1103515245U * ( (seed >> 1U) ^ (seed.yx) );
    uint  n = 1103515245U * ( (q.x) ^ (q.y >> 3U) );
    return float(n) * (1.0 / float(0xffffffffU));
}

#define N_TEXTURES 6
#define N_BOXES 2
#define N_RECTANGLES 0
#define N_SPHERES 2
#define N_MESHES 0
#define N_TRIANGLES 0
#define N_LIGHTS 0
#define N_AREA_LIGHTS 0
#define N_POINT_LIGHTS 0
#define N_SPOT_LIGHTS 0
uniform sampler2D u_TexSamplers[N_TEXTURES];
const Box boxes[N_BOXES] = Box[](Box(vec3(-0.5, -0.5, -0.5), vec3(0.5, 0.5, 0.5), Transform(mat4(3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 2, 3, -2, 1), mat4(0.333333, 0, 0, 0, 0, 0.333333, 0, 0, 0, 0, 0.333333, 0, -0.666667, -1, 0.666667, 1), mat3(0.333333, 0, 0, 0, 0.333333, 0, 0, 0, 0.333333), vec3(3, 3, 3)), 0, Material(vec3(0.9, 0.9, 1), 0, 1.8, 4, -1, -1, -1)),
Box(vec3(-0.5, -0.5, -0.5), vec3(0.5, 0.5, 0.5), Transform(mat4(2.5, 0, 0, 0, 0, 2.5, 0, 0, 0, 0, 2.5, 0, -3, 2, -2, 1), mat4(0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 1.2, -0.8, 0.8, 1), mat3(0.4, 0, 0, 0, 0.4, 0, 0, 0, 0.4), vec3(2.5, 2.5, 2.5)), 1, Material(vec3(1, 1, 1), 0, -1, 2, -1, -1, -1))
);
const Sphere spheres[N_SPHERES] = Sphere[](Sphere(vec3(0, 0, 0), 1, Transform(mat4(1.3, 0, 0, 0, 0, 1.3, 0, 0, 0, 0, 1.3, 0, -3, 1, 3, 1), mat4(0.769231, 0, 0, 0, 0, 0.769231, 0, 0, 0, 0, 0.769231, 0, 2.30769, -0.769231, -2.30769, 1), mat3(0.769231, 0, 0, 0, 0.769231, 0, 0, 0, 0.769231), vec3(1.3, 1.3, 1.3)), 2, Material(vec3(0.9, 0.9, 1), 0, 1.8, 4, -1, -1, -1)),
Sphere(vec3(0, 0, 0), 1, Transform(mat4(1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 1.5, 0, 2, 1, 3, 1), mat4(0.666667, 0, 0, 0, 0, 0.666667, 0, 0, 0, 0, 0.666667, 0, -1.33333, -0.666667, -2, 1), mat3(0.666667, 0, 0, 0, 0.666667, 0, 0, 0, 0.666667), vec3(1.5, 1.5, 1.5)), 3, Material(vec3(1, 1, 1), 0, -1, 2, -1, -1, -1))
);


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



// Computes the BSDF for a given woW and wiW
// Does not generate new samples
vec3 f_diffuse(vec3 albedo) {
    // DONE: albedo / pi = BSDF
    // pi normalizes BRDF so the total reflected light follows energy conservation
    return albedo / PI;
}

// Generates a new wiW and returns BSDF
vec3 Sample_f_diffuse(vec3 albedo, vec2 xi, vec3 nor,
                      out vec3 wiW, out float pdf, out int sampledType) {
    // DONE: // Implement cosine weighted hemisphere sampling to generate wi
    // Make sure you set wiW to a world-space ray direction,
    // since wo is in tangent space. You can use
    // the function LocalToWorld() in the "defines" file
    // to easily make a mat3 to do this conversion.
    vec3 wi = vec3(squareToHemisphereCosine(xi)); // gen (x, y, z) sample in local space
    wiW = mat3(LocalToWorld(nor)) * wi; // convert normal to world space, get final world-space incident dir
    pdf = CosTheta(wi) / PI;
    sampledType = DIFFUSE_REFL;

    return albedo / PI;
}

// Sets out vars: wiW, sampledType
// Return: BSDF
vec3 Sample_f_specular_refl(vec3 albedo, vec3 nor, vec3 wo,
                            out vec3 wiW, out int sampledType) {
    // DONE:
    // Since ωo is in tangent/local space (where the surface normal is aligned with the Z-axis),
    // the reflection about the normal is given by xyz = -x-yz
    vec3 wi = vec3(-wo.x, -wo.y, wo.z);

    // Make sure you set wiW to a world-space ray direction, since wo is in tangent/local space
    wiW = mat3(LocalToWorld(nor)) * wi;
    sampledType = SPEC_REFL;

    // for specular materials bsdf return albedo / |cos(theta)| to account for infinity bsdf
    return albedo / AbsDot(wi, nor); // divide by local space cos
}

vec3 Sample_f_specular_trans(vec3 albedo, vec3 nor, vec3 wo,
                             out vec3 wiW, out int sampledType) {
    // Hard-coded to index of refraction of glass
    float etaA = 1.;
    float etaB = 1.55;
    vec3 wi; // local space wi

    // figure out which eta is incident and which is transmitted
    bool entering = wo.z > 0.f;
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;
    vec3 localNor = entering ? vec3(0, 0, 1) : vec3(0, 0, -1);

    if (!Refract(wo, localNor, etaI / etaT, wi)) { // if refraction was successful
        // return black in the case of total internal reflection (no light is being transmitted from that point)
        return vec3(0.f);
    }

    wiW = mat3(LocalToWorld(nor)) * wi;
    sampledType = SPEC_TRANS;
    return albedo / AbsDot(wi, localNor);
}

vec3 FresnelDielectricEval(float cosThetaI) {
    // We will hard-code the indices of refraction to be
    // those of glass
    float etaI = 1.;
    float etaT = 1.55;
    cosThetaI = clamp(cosThetaI, -1.f, 1.f);

    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float temp = etaI;   // swap
        etaI = etaT;
        etaT = temp;
        cosThetaI = abs(cosThetaI);
    }

    // Compute cosThetaT using Snell's law
    float sinThetaI = sqrt(max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    // Handle total internal reflection
    if (sinThetaT >= 1.f) {
        return vec3(1.f, 1.f, 1.f);
    }
    float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));

    // Calculationss
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                    ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                    ((etaI * cosThetaI) + (etaT * cosThetaT));
    float val = ((Rparl * Rparl + Rperp * Rperp)) / 2.f;
    return vec3(val, val, val);

}

vec3 Sample_f_glass(vec3 albedo, vec3 nor, vec2 xi, vec3 wo,
                    out vec3 wiW, out int sampledType) {
    float random = rng();
    if(random < 0.5) {
        // Have to double contribution b/c we only sample
        // reflection BxDF half the time
        vec3 R = Sample_f_specular_refl(albedo, nor, wo, wiW, sampledType);
        sampledType = SPEC_REFL;
        return 2. * FresnelDielectricEval(dot(nor, normalize(wiW))) * R;
    }
    else {
        // Have to double contribution b/c we only sample
        // transmit BxDF half the time
        vec3 T = Sample_f_specular_trans(albedo, nor, wo, wiW, sampledType);
        sampledType = SPEC_TRANS;
        return 2. * (vec3(1.) - FresnelDielectricEval(dot(nor, normalize(wiW)))) * T;
    }
}

// Below are a bunch of functions for handling microfacet materials.
// Don't worry about this for now.
vec3 Sample_wh(vec3 wo, vec2 xi, float roughness) {
    vec3 wh;

    float cosTheta = 0;
    float phi = TWO_PI * xi[1];
    // We'll only handle isotropic microfacet materials
    float tanTheta2 = roughness * roughness * xi[0] / (1.0f - xi[0]);
    cosTheta = 1 / sqrt(1 + tanTheta2);

    float sinTheta =
            sqrt(max(0.f, 1.f - cosTheta * cosTheta));

    wh = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    if (!SameHemisphere(wo, wh)) wh = -wh;

    return wh;
}

float TrowbridgeReitzD(vec3 wh, float roughness) {
    float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta)) return 0.f;

    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

    float e =
            (Cos2Phi(wh) / (roughness * roughness) + Sin2Phi(wh) / (roughness * roughness)) *
            tan2Theta;
    return 1 / (PI * roughness * roughness * cos4Theta * (1 + e) * (1 + e));
}

float Lambda(vec3 w, float roughness) {
    float absTanTheta = abs(TanTheta(w));
    if (isinf(absTanTheta)) return 0.;

    // Compute alpha for direction w
    float alpha =
            sqrt(Cos2Phi(w) * roughness * roughness + Sin2Phi(w) * roughness * roughness);
    float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
    return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
}

float TrowbridgeReitzG(vec3 wo, vec3 wi, float roughness) {
    return 1 / (1 + Lambda(wo, roughness) + Lambda(wi, roughness));
}

float TrowbridgeReitzPdf(vec3 wo, vec3 wh, float roughness) {
    return TrowbridgeReitzD(wh, roughness) * AbsCosTheta(wh);
}

vec3 f_microfacet_refl(vec3 albedo, vec3 wo, vec3 wi, float roughness) {
    float cosThetaO = AbsCosTheta(wo);
    float cosThetaI = AbsCosTheta(wi);
    vec3 wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) return vec3(0.f);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return vec3(0.f);
    wh = normalize(wh);
    // TODO: Handle different Fresnel coefficients
    vec3 F = vec3(1.);//fresnel->Evaluate(glm::dot(wi, wh));
    float D = TrowbridgeReitzD(wh, roughness);
    float G = TrowbridgeReitzG(wo, wi, roughness);
    return albedo * D * G * F /
            (4 * cosThetaI * cosThetaO);
}

vec3 Sample_f_microfacet_refl(vec3 albedo, vec3 nor, vec2 xi, vec3 wo, float roughness,
                              out vec3 wiW, out float pdf, out int sampledType) {
    if (wo.z == 0) return vec3(0.);

    vec3 wh = Sample_wh(wo, xi, roughness);
    vec3 wi = reflect(-wo, wh);
    wiW = LocalToWorld(nor) * wi;
    if (!SameHemisphere(wo, wi)) return vec3(0.f);

    // Compute PDF of _wi_ for microfacet reflection
    pdf = TrowbridgeReitzPdf(wo, wh, roughness) / (4 * dot(wo, wh));
    return f_microfacet_refl(albedo, wo, wi, roughness);
}

vec3 computeAlbedo(Intersection isect) {
    vec3 albedo = isect.material.albedo;
#if N_TEXTURES
    if(isect.material.albedoTex != -1) {
        albedo *= pow(texture(u_TexSamplers[isect.material.albedoTex], isect.uv).rgb, vec3(2.2f));
    }
#endif
    return albedo;
}

vec3 computeNormal(Intersection isect) {
    vec3 nor = isect.nor;
#if N_TEXTURES
    if(isect.material.normalTex != -1) {
        vec3 localNor = texture(u_TexSamplers[isect.material.normalTex], isect.uv).rgb;
        vec3 tan, bit;
        coordinateSystem(nor, tan, bit);
        nor = mat3(tan, bit, nor) * localNor;
    }
#endif
    return nor;
}

float computeRoughness(Intersection isect) {
    float roughness = isect.material.roughness;
#if N_TEXTURES
    if(isect.material.roughnessTex != -1) {
        roughness = texture(u_TexSamplers[isect.material.roughnessTex], isect.uv).r;
    }
#endif
    return roughness;
}

// Computes the overall light scattering properties of a point on a Material,
// given the incoming and outgoing light directions.
// Direct Lighting
vec3 f(Intersection isect, vec3 woW, vec3 wiW) {
    // Convert the incoming and outgoing light rays from
    // world space to local tangent space
    vec3 nor = computeNormal(isect);
    vec3 wo = WorldToLocal(nor) * woW;
    vec3 wi = WorldToLocal(nor) * wiW;

    // If the outgoing ray is parallel to the surface,
    // we know we can return black b/c the Lambert term
    // in the overall Light Transport Equation will be 0.
    if (wo.z == 0) return vec3(0.f);

    // Since GLSL does not support classes or polymorphism,
    // we have to handle each material type with its own function.
    if(isect.material.type == DIFFUSE_REFL) {
        return f_diffuse(computeAlbedo(isect));
    }
    // As we discussed in class, there is a 0% chance that a randomly
    // chosen wi will be the perfect reflection / refraction of wo,
    // so any specular material will have a BSDF of 0 when wi is chosen
    // independently of the material.
    else if(isect.material.type == SPEC_REFL ||
            isect.material.type == SPEC_TRANS ||
            isect.material.type == SPEC_GLASS) {
        return vec3(0.);
    }
    else if(isect.material.type == MICROFACET_REFL) {
        return f_microfacet_refl(computeAlbedo(isect),
                                 wo, wi,
                                 computeRoughness(isect));
    }
    // Default case, unhandled material
    else {
        return vec3(1,0,1);
    }
}

// Sample_f() returns the same values as f(), but importantly it
// only takes in a wo. Note that wiW is declared as an "out vec3";
// this means the function is intended to compute and write a wi
// in world space (the trailing "W" indicates world space).
// In other words, Sample_f() evaluates the BSDF *after* generating
// a wi based on the Intersection's material properties, allowing
// us to bias our wi samples in a way that gives more consistent
// light scattered along wo.
// Indirect Lighting
vec3 Sample_f(Intersection isect, vec3 woW, vec2 xi,
              out vec3 wiW, out float pdf, out int sampledType) {
    // Convert wo to local space from world space.
    // The various Sample_f()s output a wi in world space,
    // but assume wo is in local space.
    vec3 nor = computeNormal(isect);
    vec3 wo = WorldToLocal(nor) * woW;

    if(isect.material.type == DIFFUSE_REFL) {
        return Sample_f_diffuse(computeAlbedo(isect), xi, nor, wiW, pdf, sampledType);
    }
    else if(isect.material.type == SPEC_REFL) {
        pdf = 1.;
        return Sample_f_specular_refl(computeAlbedo(isect), nor, wo, wiW, sampledType);
    }
    else if(isect.material.type == SPEC_TRANS) {
        pdf = 1.;
        return Sample_f_specular_trans(computeAlbedo(isect), nor, wo, wiW, sampledType);
    }
    else if(isect.material.type == SPEC_GLASS) {
        pdf = 1.;
        return Sample_f_glass(computeAlbedo(isect), nor, xi, wo, wiW, sampledType);
    }
    else if(isect.material.type == MICROFACET_REFL) {
        return Sample_f_microfacet_refl(computeAlbedo(isect),
                                        nor, xi, wo,
                                        computeRoughness(isect),
                                        wiW, pdf,
                                        sampledType);
    }
    else if(isect.material.type == PLASTIC) {
        return vec3(1,0,1);
    }
    // Default case, unhandled material
    else {
        return vec3(1,0,1);
    }
}

// Compute the PDF of wi with respect to wo and the intersection's
// material properties.
float Pdf(Intersection isect, vec3 woW, vec3 wiW) {
    vec3 nor = computeNormal(isect);
    vec3 wo = WorldToLocal(nor) * woW;
    vec3 wi = WorldToLocal(nor) * wiW;

    if (wo.z == 0) return 0.; // The cosine of this vector would be zero

    if(isect.material.type == DIFFUSE_REFL) {
        // DONE: Implement the PDF of a Lambertian material
        // in local space the surface nor is aligned with z axis
        // wi.z is cos theta / pi is the hemisphere surface area
        return CosTheta(wi) / PI;
    }
    else if(isect.material.type == SPEC_REFL ||
            isect.material.type == SPEC_TRANS ||
            isect.material.type == SPEC_GLASS) {
        return 0.;
    }
    else if(isect.material.type == MICROFACET_REFL) {
        vec3 wh = normalize(wo + wi);
        return TrowbridgeReitzPdf(wo, wh, computeRoughness(isect)) / (4 * dot(wo, wh));
    }
    // Default case, unhandled material
    else {
        return 0.;
    }
}

// optimized algorithm for solving quadratic equations developed by Dr. Po-Shen Loh -> https://youtu.be/XKBX0r3J-9Y
// Adapted to root finding (ray t0/t1) for all quadric shapes (sphere, ellipsoid, cylinder, cone, etc.) by Erich Loftis
void solveQuadratic(float A, float B, float C, out float t0, out float t1) {
    float invA = 1.0 / A;
    B *= invA;
    C *= invA;
    float neg_halfB = -B * 0.5;
    float u2 = neg_halfB * neg_halfB - C;
    float u = u2 < 0.0 ? neg_halfB = 0.0 : sqrt(u2);
    t0 = neg_halfB - u;
    t1 = neg_halfB + u;
}

vec2 sphereUVMap(vec3 p) {
    float phi = atan(p.z, p.x);
    if(phi < 0) {
        phi += TWO_PI;
    }
    float theta = acos(p.y);
    return vec2(1 - phi/TWO_PI, 1 - theta / PI);
}

float sphereIntersect(Ray ray, float radius, vec3 pos, out vec3 localNor, out vec2 out_uv, mat4 invT) {
    ray.origin = vec3(invT * vec4(ray.origin, 1.));
    ray.direction = vec3(invT * vec4(ray.direction, 0.));
    float t0, t1;
    vec3 diff = ray.origin - pos;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(ray.direction, diff);
    float c = dot(diff, diff) - (radius * radius);
    solveQuadratic(a, b, c, t0, t1);
    localNor = t0 > 0.0 ? ray.origin + t0 * ray.direction : ray.origin + t1 * ray.direction;
    localNor = normalize(localNor);
    out_uv = sphereUVMap(localNor);
    return t0 > 0.0 ? t0 : t1 > 0.0 ? t1 : INFINITY;
}

float planeIntersect( vec4 pla, vec3 rayOrigin, vec3 rayDirection, mat4 invT) {
    rayOrigin = vec3(invT * vec4(rayOrigin, 1.));
    rayDirection = vec3(invT * vec4(rayDirection, 0.));
    vec3 n = pla.xyz;
    float denom = dot(n, rayDirection);

    vec3 pOrO = (pla.w * n) - rayOrigin;
    float result = dot(pOrO, n) / denom;
    return (result > 0.0) ? result : INFINITY;
}

float rectangleIntersect(vec3 pos, vec3 normal,
                         float radiusU, float radiusV,
                         vec3 rayOrigin, vec3 rayDirection,
                         out vec2 out_uv, mat4 invT) {
    rayOrigin = vec3(invT * vec4(rayOrigin, 1.));
    rayDirection = vec3(invT * vec4(rayDirection, 0.));
    float dt = dot(-normal, rayDirection);
    // use the following for one-sided rectangle
    if (dt < 0.0) return INFINITY;
    float t = dot(-normal, pos - rayOrigin) / dt;
    if (t < 0.0) return INFINITY;

    vec3 hit = rayOrigin + rayDirection * t;
    vec3 vi = hit - pos;
    vec3 U = normalize( cross( abs(normal.y) < 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0), normal ) );
    vec3 V = cross(normal, U);

    out_uv = vec2(dot(U, vi) / length(U), dot(V, vi) / length(V));
    out_uv = out_uv + vec2(0.5, 0.5);

    return (abs(dot(U, vi)) > radiusU || abs(dot(V, vi)) > radiusV) ? INFINITY : t;
}

float boxIntersect(vec3 minCorner, vec3 maxCorner,
                   mat4 invT, mat3 invTransT,
                   vec3 rayOrigin, vec3 rayDirection,
                   out vec3 normal, out bool isRayExiting,
                   out vec2 out_uv) {
        rayOrigin = vec3(invT * vec4(rayOrigin, 1.));
        rayDirection = vec3(invT * vec4(rayDirection, 0.));
        vec3 invDir = 1.0 / rayDirection;
        vec3 near = (minCorner - rayOrigin) * invDir;
        vec3 far  = (maxCorner - rayOrigin) * invDir;
        vec3 tmin = min(near, far);
        vec3 tmax = max(near, far);
        float t0 = max( max(tmin.x, tmin.y), tmin.z);
        float t1 = min( min(tmax.x, tmax.y), tmax.z);
        if (t0 > t1) return INFINITY;
        if (t0 > 0.0) // if we are outside the box
        {
                normal = -sign(rayDirection) * step(tmin.yzx, tmin) * step(tmin.zxy, tmin);
                normal = normalize(invTransT * normal);
                isRayExiting = false;
                vec3 p = t0 * rayDirection + rayOrigin;
                p = (p - minCorner) / (maxCorner - minCorner);
                out_uv = p.xy;
                return t0;
        }
        if (t1 > 0.0) // if we are inside the box
        {
                normal = -sign(rayDirection) * step(tmax, tmax.yzx) * step(tmax, tmax.zxy);
                normal = normalize(invTransT * normal);
                isRayExiting = true;
                vec3 p = t1 * rayDirection + rayOrigin;
                p = (p - minCorner) / (maxCorner - minCorner);
                out_uv = p.xy;
                return t1;
        }
        return INFINITY;
}

// Möller–Trumbore intersection
float triangleIntersect(vec3 p0, vec3 p1, vec3 p2,
                        vec3 rayOrigin, vec3 rayDirection) {
    const float EPSILON = 0.0000001;
    vec3 edge1, edge2, h, s, q;
    float a,f,u,v;
    edge1 = p1 - p0;
    edge2 = p2 - p0;
    h = cross(rayDirection, edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        return INFINITY;    // This ray is parallel to this triangle.
    }
    f = 1.0/a;
    s = rayOrigin - p0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return INFINITY;
    q = cross(s, edge1);
    v = f * dot(rayDirection, q);
    if (v < 0.0 || u + v > 1.0) {
        return INFINITY;
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * dot(edge2, q);
    if (t > EPSILON) {
        return t;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return INFINITY;
}

vec3 barycentric(vec3 p, vec3 t1, vec3 t2, vec3 t3) {
    vec3 edge1 = t2 - t1;
    vec3 edge2 = t3 - t2;
    float S = length(cross(edge1, edge2));

    edge1 = p - t2;
    edge2 = p - t3;
    float S1 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t3;
    float S2 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t2;
    float S3 = length(cross(edge1, edge2));

    return vec3(S1 / S, S2 / S, S3 / S);
}

#if N_MESHES
float meshIntersect(int mesh_id,
                    vec3 rayOrigin, vec3 rayDirection,
                    out vec3 out_nor, out vec2 out_uv,
                    mat4 invT) {

    rayOrigin = vec3(invT * vec4(rayOrigin, 1.));
    rayDirection = vec3(invT * vec4(rayDirection, 0.));

    int sampIdx = 0;// meshes[mesh_id].triangle_sampler_index;

    float t = INFINITY;

    // Iterate over each triangle, and
    // convert it to a pixel coordinate
    for(int i = 0; i < meshes[mesh_id].num_tris; ++i) {
        // pos0, pos1, pos2, nor0, nor1, nor2, uv0, uv1, uv2
        // Each triangle takes up 9 pixels
        Triangle tri;
        int first_pixel = i * 9;
        // Positions
        for(int p = first_pixel; p < first_pixel + 3; ++p) {
            int row = int(floor(float(p) / meshes[mesh_id].triangle_storage_side_len));
            int col = p - row * meshes[mesh_id].triangle_storage_side_len;

            tri.pos[p - first_pixel] = texelFetch(u_TriangleStorageSamplers[sampIdx],
                                                ivec2(col, row), 0).rgb;
        }
        first_pixel += 3;
        // Normals
        for(int n = first_pixel; n < first_pixel + 3; ++n) {
            int row = int(floor(float(n) / meshes[mesh_id].triangle_storage_side_len));
            int col = n - row * meshes[mesh_id].triangle_storage_side_len;

            tri.nor[n - first_pixel] = texelFetch(u_TriangleStorageSamplers[sampIdx],
                                                ivec2(col, row), 0).rgb;
        }
        first_pixel += 3;
        // UVs
        for(int v = first_pixel; v < first_pixel + 3; ++v) {
            int row = int(floor(float(v) / meshes[mesh_id].triangle_storage_side_len));
            int col = v - row * meshes[mesh_id].triangle_storage_side_len;

            tri.uv[v - first_pixel] = texelFetch(u_TriangleStorageSamplers[sampIdx],
                                               ivec2(col, row), 0).rg;
        }

        float d = triangleIntersect(tri.pos[0], tri.pos[1], tri.pos[2],
                                    rayOrigin, rayDirection);
        if(d < t) {
            t = d;
            vec3 p = rayOrigin + t * rayDirection;
            vec3 baryWeights = barycentric(p, tri.pos[0], tri.pos[1], tri.pos[2]);
            out_nor = baryWeights[0] * tri.nor[0] +
                      baryWeights[1] * tri.nor[1] +
                      baryWeights[2] * tri.nor[2];
            out_uv =  baryWeights[0] * tri.uv[0] +
                      baryWeights[1] * tri.uv[1] +
                      baryWeights[2] * tri.uv[2];
        }
    }

    return t;
}
#endif

Intersection sceneIntersect(Ray ray) {
    float t = INFINITY;
    Intersection result;
    result.t = INFINITY;

#if N_RECTANGLES
    for(int i = 0; i < N_RECTANGLES; ++i) {
        vec2 uv;
        Rectangle rect = rectangles[i];
        float d = rectangleIntersect(rect.pos, rect.nor,
                                     rect.halfSideLengths.x,
                                     rect.halfSideLengths.y,
                                     ray.origin, ray.direction,
                                     uv,
                                     rect.transform.invT);
        if(d < t) {
            t = d;
            result.t = t;
            result.nor = normalize(rect.transform.invTransT * rect.nor);
            result.uv = uv;
            result.Le = vec3(0,0,0);
            result.obj_ID = rect.ID;
            result.material = rect.material;
        }
    }
#endif
#if N_BOXES
    for(int i = 0; i < N_BOXES; ++i) {
        vec3 nor;
        bool isExiting;
        vec2 uv;
        Box b = boxes[i];
        float d = boxIntersect(b.minCorner, b.maxCorner,
                               b.transform.invT, b.transform.invTransT,
                               ray.origin, ray.direction,
                               nor, isExiting, uv);
        if(d < t) {
            t = d;
            result.t = t;
            result.nor = nor;
            result.Le = vec3(0,0,0);
            result.obj_ID = b.ID;
            result.material = b.material;
            result.uv = uv;
        }
    }
#endif
#if N_SPHERES
    for(int i = 0; i < N_SPHERES; ++i) {
        vec3 nor;
        bool isExiting;
        vec3 localNor;
        vec2 uv;
        Sphere s = spheres[i];
        float d = sphereIntersect(ray, s.radius, s.pos, localNor, uv,
                                  s.transform.invT);
        if(d < t) {
            t = d;
            vec3 p = ray.origin + t * ray.direction;
            result.t = t;
            result.nor = normalize(s.transform.invTransT * localNor);
            result.Le = vec3(0,0,0);
            result.uv = uv;
            result.obj_ID = s.ID;
            result.material = s.material;
        }
    }
#endif
#if N_MESHES
    for(int i = 0; i < N_MESHES; ++i) {
        vec3 nor;
        vec2 uv;
        float d = meshIntersect(i, ray.origin, ray.direction,
                                nor, uv, meshes[i].transform.invT);

        if(d < t) {
            t = d;
            result.t = t;
            result.nor = nor;
            result.uv =  uv;
            result.Le = vec3(0,0,0);
            result.obj_ID = meshes[i].ID;
            result.material = meshes[i].material;
        }
    }
#endif
#if N_AREA_LIGHTS
    for(int i = 0; i < N_AREA_LIGHTS; ++i) {
        AreaLight l = areaLights[i];
        int shapeType = l.shapeType;
        if(shapeType == RECTANGLE) {
            vec3 pos = vec3(0,0,0);
            vec3 nor = vec3(0,0,1);
            vec2 halfSideLengths = vec2(0.5, 0.5);
            vec2 uv;
            float d = rectangleIntersect(pos, nor,
                                   halfSideLengths.x,
                                   halfSideLengths.y,
                                   ray.origin, ray.direction,
                                   uv,
                                   l.transform.invT);
            if(d < t) {
                t = d;
                result.t = t;
                result.nor = normalize(l.transform.invTransT * vec3(0,0,1));
                result.Le = l.Le;
                result.obj_ID = l.ID;
            }
        }
        else if(shapeType == SPHERE) {
            vec3 pos = vec3(0,0,0);
            float radius = 1.;
            mat4 invT = l.transform.invT;
            vec3 localNor;
            vec2 uv;
            float d = sphereIntersect(ray, radius, pos, localNor, uv, invT);
            if(d < t) {
                t = d;
                result.t = t;
                result.nor = normalize(l.transform.invTransT * localNor);
                result.Le = l.Le;
                result.obj_ID = l.ID;
            }
        }
    }
#endif
#if N_TEXTURES
    if(result.material.normalTex != -1) {
        vec3 localNor = texture(u_TexSamplers[result.material.normalTex], result.uv).rgb;
        localNor = localNor * 2. - vec3(1.);
        vec3 tan, bit;
        coordinateSystem(result.nor, tan, bit);
        result.nor = mat3(tan, bit, result.nor) * localNor;
    }
#endif
    return result;
}

Intersection areaLightIntersect(AreaLight light, Ray ray) {
    Intersection result;
    result.t = INFINITY;
#if N_AREA_LIGHTS
    int shapeType = light.shapeType;
    if(shapeType == RECTANGLE) {
        vec3 pos = vec3(0,0,0);
        vec3 nor = vec3(0,0,1);
        vec2 halfSideLengths = vec2(0.5, 0.5);
        vec2 uv;
        float d = rectangleIntersect(pos, nor,
                               halfSideLengths.x,
                               halfSideLengths.y,
                               ray.origin, ray.direction,
                               uv,
                               light.transform.invT);
        result.t = d;
        result.nor = normalize(light.transform.invTransT * vec3(0,0,1));
        result.Le = light.Le;
        result.obj_ID = light.ID;
    }
    else if(shapeType == SPHERE) {
        vec3 pos = vec3(0,0,0);
        float radius = 1.;
        mat4 invT = light.transform.invT;
        vec3 localNor;
        vec2 uv;
        float d = sphereIntersect(ray, radius, pos, localNor, uv, invT);
        result.t = d;
        result.nor = normalize(light.transform.invTransT * localNor);
        result.Le = light.Le;
        result.obj_ID = light.ID;
    }
#endif
    return result;
}

vec2 normalize_uv = vec2(0.1591, 0.3183);
vec2 sampleSphericalMap(vec3 v) {
    // U is in the range [-PI, PI], V is [-PI/2, PI/2]
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    // Convert UV to [-0.5, 0.5] in U&V
    uv *= normalize_uv;
    // Convert UV to [0, 1]
    uv += 0.5;
    return uv;
}

vec3 sampleFromInsideSphere(vec2 xi, out float pdf) {
//    Point3f pObj = WarpFunctions::squareToSphereUniform(xi);

//    Intersection it;
//    it.normalGeometric = glm::normalize( transform.invTransT() *pObj );
//    it.point = Point3f(transform.T() * glm::vec4(pObj.x, pObj.y, pObj.z, 1.0f));

//    *pdf = 1.0f / Area();

//    return it;
    return vec3(0.);
}

#if N_AREA_LIGHTS
vec3 DirectSampleAreaLight(int idx,
                           vec3 view_point, vec3 view_nor,
                           int num_lights,
                           out vec3 wiW, out float pdf) {
    AreaLight light = areaLights[idx];
    int type = light.shapeType;
    Ray shadowRay;

    if(type == RECTANGLE) {
        // Compute a random point on the surface of light,
        // making use of the light source's Transform and the rng function
        // You may assume that an untransformed rectangular area light has a
        // surface normal of (0,0,1), a side length of 2, and is centered at the origin
        // (i.e. its X and Y coords span the range [-1, 1]).
        vec3 lightLocalPos = vec3(rng() * 2.f - 1.f, rng() * 2.f - 1.f, 0.f); // [-1, 1]
        vec3 lightPos = (light.transform.T * vec4(lightLocalPos, 1)).xyz; // local to world
        vec3 lightNor = normalize(light.transform.invTransT * vec3(0, 0, 1)); // local to world

        // Set ωi to the normalized vector from the reference point to the generated light source point
        wiW = normalize(lightPos - view_point);

        // Compute the PDF of the chosen point with respect to the area of the AreaLight
        // Convert the PDF to be with respect to the light source's solid angle projection on view_point
        // square distance from viewpoint to sampledLightPoint / SA * abscos(angle bw light normal and dir to sample point)
        float d2 = distance(lightPos, view_point) * distance(lightPos, view_point);
        float cosTheta = AbsDot(lightNor, wiW);
        float lightArea = 4.f * light.transform.scale.x * light.transform.scale.y;
        pdf = d2 / (lightArea * cosTheta);

        // Check to see if ωi reaches the light source, or if it intersects another object along the way
        // If there is an occluder, return black.
        Ray feelerRay = SpawnRay(view_point, wiW);
        Intersection feelerIsect = sceneIntersect(feelerRay);
        if (feelerIsect.obj_ID == light.ID) {
            // Return the light emitted along ωi, making sure to scale it by num_lights
            // to account for the fact that a given light source is only sampled 1/num_lights times on average
            return (light.Le) * num_lights;
        }

        // Return black if occluded
        return vec3(0.f);
    }
    else if(type == SPHERE) {
        Transform tr = areaLights[idx].transform;

        vec2 xi = vec2(rng(), rng());

        vec3 center = vec3(tr.T * vec4(0., 0., 0., 1.));
        vec3 centerToRef = normalize(center - view_point);
        vec3 tan, bit;

        coordinateSystem(centerToRef, tan, bit);

        vec3 pOrigin;
        if(dot(center - view_point, view_nor) > 0) {
            pOrigin = view_point + view_nor * RayEpsilon;
        }
        else {
            pOrigin = view_point - view_nor * RayEpsilon;
        }

        // Inside the sphere
        if(dot(pOrigin - center, pOrigin - center) <= 1.f) // Radius is 1, so r^2 is also 1
            return sampleFromInsideSphere(xi, pdf);

        float sinThetaMax2 = 1 / dot(view_point - center, view_point - center); // Again, radius is 1
        float cosThetaMax = sqrt(max(0.0f, 1.0f - sinThetaMax2));
        float cosTheta = (1.0f - xi.x) + xi.x * cosThetaMax;
        float sinTheta = sqrt(max(0.f, 1.0f- cosTheta * cosTheta));
        float phi = xi.y * TWO_PI;

        float dc = distance(view_point, center);
        float ds = dc * cosTheta - sqrt(max(0.0f, 1 - dc * dc * sinTheta * sinTheta));

        float cosAlpha = (dc * dc + 1 - ds * ds) / (2 * dc * 1);
        float sinAlpha = sqrt(max(0.0f, 1.0f - cosAlpha * cosAlpha));

        vec3 nObj = sinAlpha * cos(phi) * -tan + sinAlpha * sin(phi) * -bit + cosAlpha * -centerToRef;
        vec3 pObj = vec3(nObj); // Would multiply by radius, but it is always 1 in object space

        shadowRay = SpawnRay(view_point, normalize(vec3(tr.T * vec4(pObj, 1.0f)) - view_point));
        wiW = shadowRay.direction;
        pdf = 1.0f / (TWO_PI * (1 - cosThetaMax));
        pdf /= tr.scale.x * tr.scale.x;
    }

    Intersection isect = sceneIntersect(shadowRay);
    if(isect.obj_ID == areaLights[idx].ID) {
        // Multiply by N+1 to account for sampling it 1/(N+1) times.
        // +1 because there's also the environment light
        return num_lights * areaLights[idx].Le;
    }
}
#endif

#if N_POINT_LIGHTS
vec3 DirectSamplePointLight(int idx,
                            vec3 view_point, int num_lights,
                            out vec3 wiW, out float pdf) {
    PointLight light = pointLights[idx];
    // Generate ωi by normalizing a vector from view point to light source
    wiW = normalize(light.pos - view_point);

    // Set the PDF equal to the nonzero value of a Dirac Delta Distribution,
    // since our ωi is guaranteed to go towards our light
    pdf = 1.0f;

    // Find the intersection of a shadow feeler ray with your scene.
    // If the point of intersection has a t value greater than the distance
    // between view_point and the light, then the light is NOT occluded.
    Ray feelerRay = SpawnRay(view_point, wiW);
    Intersection feelerIsect = sceneIntersect(feelerRay);
    if (feelerIsect.t > abs(distance(light.pos, view_point))) {
        // Return the light's Le divided by the squared distance between it and view_point if not occluded
        // making sure to scale it by num_lights to account for the fact that a
        // given light source is only sampled 1/num_lights times on average.
        return (light.Le / (distance(light.pos, view_point) * distance(light.pos, view_point))) * num_lights;
    }

    // Return black if occluded
    return vec3(0.f);
}
#endif

#if N_SPOT_LIGHTS
vec3 DirectSampleSpotLight(int idx,
                           vec3 view_point, int num_lights,
                           out vec3 wiW, out float pdf) {
    SpotLight light = spotLights[idx];
    // spotlight is assumed to be at the origin, facing (0, 0, 1) before transformation
    vec3 forward = normalize(light.transform.invTransT * vec3(0, 0, 1));
    vec3 pos = (light.transform.T * vec4(0, 0, 0, 1)).xyz;
    wiW = normalize(pos - view_point);
    pdf = 1.0f;

    // Only view_points that lie within the spot light's outerAngle receive light energy
    // Compute the angle between the spotlight's direction and the light-to-point direction
    float cosTheta = dot(forward, -wiW);
    float cosFalloffStart = cos(radians(light.innerAngle));
    float cosTotalWidth = cos(radians(light.outerAngle));

    Ray feelerRay = SpawnRay(view_point, wiW);
    Intersection feelerIsect = sceneIntersect(feelerRay);
    if (feelerIsect.t > abs(distance(pos, view_point))) {
        // use smoothstep for cubic interpolation (spotlight's soft edge)
        // view_points that lie between the spot light's innerAngle and outerAngle
        // receive reduced light energy. The amount of reduction is a cubic falloff
        // use smoothstep with the inner and outer angles as its edge arguments,
        // and view_point's relative angle as its x argument.
        float intensity = smoothstep(cosTotalWidth, cosFalloffStart, cosTheta);
        return (light.Le / (distance(pos, view_point) * distance(pos, view_point))) * intensity * num_lights;
    }

    // Return black if occluded
    return vec3(0.f);
}
#endif



vec3 Sample_Li(vec3 view_point, vec3 nor, out vec3 wiW, out float pdf) {
    // Choose a random light from among all of the
    // light sources in the scene, including the environment light
    int num_lights = N_LIGHTS;

#define ENV_MAP1 0
#if ENV_MAP1
    int num_lights = N_LIGHTS + 1;
#endif
    int randomLightIdx = int(rng() * num_lights);

    // Chose an area light
    if(randomLightIdx < N_AREA_LIGHTS) {
#if N_AREA_LIGHTS
        return DirectSampleAreaLight(randomLightIdx, view_point, nor, num_lights,
                                     wiW, pdf);
#endif
    }
    // Chose a point light
    else if(randomLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS) {
#if N_POINT_LIGHTS
        return DirectSamplePointLight(randomLightIdx - N_AREA_LIGHTS,
                                      view_point, num_lights, wiW, pdf);
#endif
    }
    // Chose a spot light
    else if(randomLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS + N_SPOT_LIGHTS) {
#if N_SPOT_LIGHTS
        return DirectSampleSpotLight(randomLightIdx - N_AREA_LIGHTS - N_POINT_LIGHTS,
                                     view_point, num_lights, wiW, pdf);
#endif
    }
    // Chose the environment light
    else {
        // TODO
    }
    return vec3(0.);
}

vec3 Sample_Li(vec3 view_point, vec3 nor,
                       out vec3 wiW, out float pdf,
                       out int chosenLightIdx,
                       out int chosenLightID,
                       out int chosenLightType) {
    // Choose a random light from among all of the
    // light sources in the scene, including the environment light
    int num_lights = N_LIGHTS;
#define ENV_MAP2 1
#if ENV_MAP2
    num_lights = N_LIGHTS + 1;
#endif
    int randomLightIdx = int(rng() * num_lights);
    chosenLightIdx = randomLightIdx;
    // Chose an area light
    if(randomLightIdx < N_AREA_LIGHTS) {
#if N_AREA_LIGHTS
        chosenLightID = areaLights[chosenLightIdx].ID;
        chosenLightType = AREA_LIGHT;
        return DirectSampleAreaLight(randomLightIdx, view_point, nor, num_lights, wiW, pdf);
#endif
    }
    // Chose a point light
    else if(randomLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS) {
#if N_POINT_LIGHTS
        chosenLightID = pointLights[randomLightIdx - N_AREA_LIGHTS].ID;
        chosenLightType = POINT_LIGHT;
        return DirectSamplePointLight(randomLightIdx - N_AREA_LIGHTS, view_point, num_lights, wiW, pdf);
#endif
    }
    // Chose a spot light
    else if(randomLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS + N_SPOT_LIGHTS) {
#if N_SPOT_LIGHTS
        chosenLightID = spotLights[randomLightIdx - N_AREA_LIGHTS - N_POINT_LIGHTS].ID;
        chosenLightType = SPOT_LIGHT;
        return DirectSampleSpotLight(randomLightIdx - N_AREA_LIGHTS - N_POINT_LIGHTS, view_point, num_lights, wiW, pdf);
#endif
    }
    // Chose the environment light
    else {
        chosenLightID = -1;
        chosenLightType = ENVIRONMENT_LIGHT;
        // TODO: allow the environment map accessed via the u_EnvironmentMap sampler2D to act as a light source

        // cosine-sample the hemisphere at isect (view_point)
        vec2 xi = vec2(rng(), rng());
        vec3 wi = vec3(squareToHemisphereCosine(xi));
        wiW = mat3(LocalToWorld(nor)) * wi;
        pdf = squareToHemisphereCosinePDF(wi);

        // perform a shadow test to make sure ωi can actually see the environment map
        // Since the environment map is intended to fill the space where no objects exist
        // ωi sees the environment map only if it hits nothing in your scene
        Ray feelerRay = SpawnRay(view_point, wiW);
        Intersection feelerIsect = sceneIntersect(feelerRay);
        if (feelerIsect.t == INFINITY) {
            // return the map's color at the UV coordinates
            // determined by the sampleSphericalMap function provided in
            // pathtracer.defines.glsl.
            vec2 uv = sampleSphericalMap(wi);
            return texture(u_EnvironmentMap, uv).rgb;
        }
    }
    return vec3(0.);
}

float UniformConePdf(float cosThetaMax) {
    return 1 / (2 * PI * (1 - cosThetaMax));
}

float SpherePdf(vec3 view_point, vec3 view_nor, vec3 p, vec3 wi,
                Transform transform, float radius) {
    vec3 pCenter = (transform.T * vec4(0, 0, 0, 1)).xyz;
    // Return uniform PDF if point is inside sphere
    vec3 pOrigin = p + view_nor * 0.0001;
    // If inside the sphere
    if(DistanceSquared(pOrigin, pCenter) <= radius * radius) {
//        return Shape::Pdf(ref, wi);
        // To be provided later
        return 0.f;
    }
    // Compute general sphere PDF
//    float sinThetaMax2 = radius * radius / DistanceSquared(p, pCenter);
    float sinThetaMax2 = 1 / dot(view_point - pCenter, view_point - pCenter); // Again, radius is 1
    float cosThetaMax = sqrt(max(0.f, 1.f - sinThetaMax2));
    return UniformConePdf(cosThetaMax) / transform.scale.x * transform.scale.x;
}

float Pdf_Li(vec3 view_point, vec3 nor, vec3 wiW, int chosenLightIdx) {
    Ray ray = SpawnRay(view_point, wiW);
    // Area light
    if(chosenLightIdx < N_AREA_LIGHTS) {
#if N_AREA_LIGHTS
        Intersection isect = areaLightIntersect(areaLights[chosenLightIdx],
                                                ray);
        if(isect.t == INFINITY) {
            return 0.;
        }
        vec3 light_point = ray.origin + isect.t * wiW;
        // If doesn't intersect, 0 PDF
        if(isect.t == INFINITY) {
            return 0.;
        }
        int type = areaLights[chosenLightIdx].shapeType;
        if(type == RECTANGLE) {
            Transform tr = areaLights[chosenLightIdx].transform;
            vec3 pos = vec3(tr.T * vec4(0,0,0,1));
            // Technically half side len
            vec2 sideLen = tr.scale.xy;
            vec3 nor = normalize(tr.invTransT * vec3(0,0,1));
            // Convert PDF from w/r/t surface area to w/r/t solid angle
            float r2 = isect.t * isect.t;
            // r*r / (cos(theta_w) * area)
            return r2 / (AbsDot(ray.direction, nor) * 4 * sideLen.x * sideLen.y);
        }
        else if(type == SPHERE) {
            return SpherePdf(view_point, isect.nor, light_point, wiW,
                                  areaLights[chosenLightIdx].transform,
                                  1.f);
        }
#endif
    }
    // Point light or spot light
    else if(chosenLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS ||
            chosenLightIdx < N_AREA_LIGHTS + N_POINT_LIGHTS + N_SPOT_LIGHTS) {
        return 0;
    }
    // Env map
    else {
        vec3 wi = WorldToLocal(nor) * wiW;
        return squareToHemisphereCosinePDF(wi);
    }
}

// The power heuristic is a modification of the balance heuristic that reduces variance
// To be used in Li_DirectMIS
float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    // DONE
    // f(x) = bsdf()   g(x) = Li()
    // balance: w(x) = nf * pdf_bsdf(wi_bsdf) / pdf_bsdf(wi_bsdf) + pdf_light(wi_bsdf)
    // power  : square the numerator and both halves of the denominator of the balance heuristic
    return pow((nf * fPdf), 2) / (pow(nf * fPdf, 2) + pow(ng * gPdf, 2));
}

// returns the result of computing the MIS direct light at
// a particular intersection in the scene
vec3 isectMIS(Intersection isect, vec3 woW, Ray ray) {
    vec3 Lo = vec3(0.f, 0.f, 0.f);
    vec3 view_point = ray.origin + isect.t * ray.direction;
    int chosenLightIdx, chosenLightID, chosenLightType;

    // ========== Light-Sampled Ray ==========
    // wiY = wi sampled w/r/t light
    // sumG: 1/nSum f(y) * g(y) * w(y) / p(y)
    vec3 wiY;
    vec3 sumG = vec3(0.f);
    float pdfLiLi;                                                // The PDF of your light-sampled ray with respect to that light = pdf_Li(wi_Li)
    vec3 Li = Sample_Li(view_point, isect.nor, wiY, pdfLiLi,
                        chosenLightIdx, chosenLightID, chosenLightType);           // Li(p, wi_light)
    if (pdfLiLi > 0.f && any(greaterThan(Li, vec3(0.f)))) {
        vec3 bsdf_Li = max(f(isect, woW, wiY), vec3(0.f));        // bsdf(p, wo, wi_light)
        float pdfBsdfLi = Pdf(isect, woW, wiY);                   // The PDF of your light-sampled ray with respect to your intersection's BSDF = pdf_bsdf(wi_Li)
        float wy = PowerHeuristic(1, pdfLiLi, 1, pdfBsdfLi);    // power: w(y) = ng * pdf_light(wi_light) / ng * pdf_light(wi_light) + ng * pdf_bsdf(wi_light)

        // Check if light-sampled ray is unoccluded
        Ray shadowRay = SpawnRay(view_point, wiY);
        Intersection shadowIsect = sceneIntersect(shadowRay);
        if (shadowIsect.obj_ID == chosenLightID || chosenLightType == SPOT_LIGHT || chosenLightType == POINT_LIGHT) {
            sumG = (bsdf_Li * Li * wy) / pdfLiLi;                 // sumG: bsdf(p, wo, wi_light) * Li(p, wi_light) * w(y) / pdf_light
        }
    }

    if (chosenLightType == POINT_LIGHT || chosenLightType == SPOT_LIGHT) {
        Lo += sumG;
        return Lo;
    }

    // ========== BSDF-Sampled Ray ==========
    // wiX = wi sampled w/r/t bsdf
    // sumF: 1/nSum f(x) * g(x) * w(x) / p(x)
    vec3 wiX;
    vec3 sumF = vec3(0.f);
    float pdfBsdfBsdf;                                                          // The PDF of your BSDF-sampled ray with respect to your intersection's BSDF = pdf_bsdf(wi_bsdf)
    vec2 xi = vec2(rng(), rng());
    int sampledType;
    vec3 bsdfBsdf = Sample_f(isect, woW, xi, wiX, pdfBsdfBsdf, sampledType);    // bsdf(p, wo, wi_bsdf)
    if (pdfBsdfBsdf > 0.f) {
        float pdfLiBsdf = Pdf_Li(view_point, isect.nor, wiX, chosenLightIdx);   // The PDF of your BSDF-sampled ray with respect to the light you chose with your light source sampling = pdf_Li(wi_bsdf)
        float wx = PowerHeuristic(1, pdfBsdfBsdf, 1, pdfLiBsdf);              // power: w(x) = nf * pdf_bsdf(wi_bsdf) / nf * pdf_bsdf(wi_bsdf) + ng * pdf_light(wi_bsdf)

        // Trace ray from BSDF sample
        Ray bsdfRay = SpawnRay(view_point, wiX);
        Intersection lightIsect = sceneIntersect(bsdfRay);
        // Accumulate light only if it hits the same light source
        if (lightIsect.obj_ID == chosenLightID) {
            vec3 Li_bsdf = lightIsect.Le;
            sumF = (bsdfBsdf * Li_bsdf * wx) / pdfBsdfBsdf;                     // sumF: bsdf(p, wo, wi_bsdf) * Li(p, wi_bsdf) * w(x) / pdf_bsdf
        }
    }

    if (any(isnan(Lo))) {
        Lo = vec3(0, 0, 0); // Set invalid colors to black
    }
    if (any(isnan(sumG))) {
        sumG = vec3(1, 0, 0); // Set invalid colors to black
    }
    if (any(isnan(sumF))) {
        sumF = vec3(0, 1, 0); // Set invalid colors to black
    }

    // average them together to produce an overall sample color
    Lo += (sumG + sumF);
    return Lo;
}


const float FOVY = 19.5f * PI / 180.0;


Ray rayCast() {
    vec2 offset = vec2(rng(), rng());
    vec2 ndc = (vec2(gl_FragCoord.xy) + offset) / vec2(u_ScreenDims);
    ndc = ndc * 2.f - vec2(1.f);

    float aspect = u_ScreenDims.x / u_ScreenDims.y;
    vec3 ref = u_Eye + u_Forward;
    vec3 V = u_Up * tan(FOVY * 0.5);
    vec3 H = u_Right * tan(FOVY * 0.5) * aspect;
    vec3 p = ref + H * ndc.x + V * ndc.y;

    return SpawnRay(u_Eye, normalize(p - u_Eye));
}

// DONE: Implement naive integration
// Depending on the stage of recursion,
// this func represents Lo or Li of the LTE
// Lo for currIter, Li for prevIter
vec3 Li_Naive(Ray ray) {

    // L_o(p, wo) (thisIterationColor)
    vec3 Lo = vec3(0.f);            // accumulated ray's light energy
    vec3 throughput = vec3(1.f);    // multiplicitiavely accumulated surface color attenuation

    for (int bounce = 0; bounce < MAX_DEPTH; ++bounce) { // bounces ray through the scene 10 times max

        // BASE CASE: No Intersection- return black
        Intersection isect = sceneIntersect(ray);
        if (isect.t == INFINITY) {
            break;
        }

        // BASE CASE: Hit a light source
        if (any(greaterThan(isect.Le, vec3(0.f)))) {
            // L_e(p, wo) (surface emission)
            Lo += isect.Le * throughput;    // complete LTE
            break;                          // Stop tracing if we hit an emissive object (light source)
        }

        // LTE: hit a non light object, evaluate LTE
        // 1/n * summation (n is a constantly increasing value over time, on our third frame n=3)
        // 1/n is dealt with in mix function in main()
        vec3 Le = vec3(0);              // not a light
        vec3 woW = -ray.direction;      // computed from ray.dir
        vec2 xi = vec2(rng(), rng());   // random sample for hemisphere sampling
        vec3 wiW;                       // incoming sampled direction, written to by Sample_f()
        float pdf;                      // the probability of choosing wiW, computed and written to inside Sample_f()
        int sampledType;                // to be written to in Sample_f

        // bsdf(p, wo, wi) (represents material properties only at point p,
        // tells us how much light will leave wo based on material (p) and existing light (wi))
        vec3 bsdf = Sample_f(isect, woW, xi, wiW, pdf, sampledType);

        // absdot(wi, nor) (lamberts law of cosines)
        float lambert = AbsDot(wiW, isect.nor);

        // Li(p, wi) (color and brightness of light)
        if (pdf > 0.f) {
            // Update throughput for future bounces to work around glsl's lack of recursion
            throughput *= bsdf * lambert / pdf;
        }

        ray = SpawnRay((ray.origin + isect.t * ray.direction), wiW);
    }

    return Lo;
}

// performs light source importance sampling and evaluates
// the light energy that a given point receives directly from light sources
vec3 Li_Direct_Simple(Ray ray) {
    // L_o(p, wo) (thisIterationColor)
    vec3 Lo = vec3(0.f, 0.f, 0.f);            // accumulated ray's light energy

    // BASE CASE: No Intersection- return black
    Intersection isect = sceneIntersect(ray);
    if (isect.t == INFINITY) {
        return Lo;
    }

    // BASE CASE: Hit a light source
    if (any(greaterThan(isect.Le, vec3(0.f)))) {
        // L_e(p, wo) (surface emission)
        Lo += isect.Le;
        return Lo;                          // Stop tracing if we hit an emissive object (light source)
    }

    // LTE: hit a non light object, evaluate LTE
    // 1/n * summation (n is a constantly increasing value over time, on our third frame n=3)
    // 1/n is dealt with in mix function in main()
    vec3 woW = -ray.direction;      // computed from ray.dir
    vec3 wiW;                       // incoming sampled direction, written to by Sample_f()
    float pdf;                      // the probability of choosing wiW, computed and written to inside Sample_f()
    vec3 view_point = ray.origin + isect.t * ray.direction;

    // Sample a light source, sets wiW and pdf
    vec3 Li = Sample_Li(view_point, isect.nor, wiW, pdf);

    vec3 bsdf = f(isect, woW, wiW);
    //vec3 bsdf = vec3(1, 0, 1);

    // absdot(wi, nor) (lamberts law of cosines)
    float lambert = AbsDot(wiW, isect.nor);

    // Li(p, wi) (color and brightness of light)
    if (pdf > 0.f) {
        Lo += (bsdf * Li * lambert) / pdf;
    }
    ray = SpawnRay((ray.origin + isect.t * ray.direction), wiW);

    return Lo;
}

// an integrator that implements multiple importance sampling to more effectively
// estimate the direct illumination within a scene containing light sources of
// different sizes and materials of different glossiness
vec3 Li_DirectMIS(Ray ray) {

    vec3 Lo = vec3(0.f, 0.f, 0.f);
    Intersection isect = sceneIntersect(ray);

    // No Intersection- return black
    if (isect.t == INFINITY) {
        return Lo;
    }

    // Hit a light source
    if (any(greaterThan(isect.Le, vec3(0.f)))) {
        Lo += isect.Le;
        return Lo;
    }

    vec3 view_point = ray.origin + isect.t * ray.direction;
    vec3 woW = -ray.direction;
    int chosenLightIdx, chosenLightID, chosenLightType;

    // ========== Light-Sampled Ray ==========
    // wiY = wi sampled w/r/t light
    // sumG: 1/nSum f(y) * g(y) * w(y) / p(y)
    vec3 wiY;
    vec3 sumG = vec3(0.f);
    float pdfLiLi;                                                // The PDF of your light-sampled ray with respect to that light = pdf_Li(wi_Li)
    vec3 Li = Sample_Li(view_point, isect.nor, wiY, pdfLiLi,
                        chosenLightIdx, chosenLightID, chosenLightType);           // Li(p, wi_light)
    if (pdfLiLi > 0.f && any(greaterThan(Li, vec3(0.f)))) {
        vec3 bsdf_Li = max(f(isect, woW, wiY), vec3(0.f));        // bsdf(p, wo, wi_light)
        float pdfBsdfLi = Pdf(isect, woW, wiY);                   // The PDF of your light-sampled ray with respect to your intersection's BSDF = pdf_bsdf(wi_Li)
        float wy = PowerHeuristic(1, pdfLiLi, 1, pdfBsdfLi);    // power: w(y) = ng * pdf_light(wi_light) / ng * pdf_light(wi_light) + ng * pdf_bsdf(wi_light)

        // Check if light-sampled ray is unoccluded
        Ray shadowRay = SpawnRay(view_point, wiY);
        Intersection shadowIsect = sceneIntersect(shadowRay);
        if (shadowIsect.obj_ID == chosenLightID || chosenLightType == SPOT_LIGHT || chosenLightType == POINT_LIGHT) {
            sumG = (bsdf_Li * Li * wy) / pdfLiLi;                 // sumG: bsdf(p, wo, wi_light) * Li(p, wi_light) * w(y) / pdf_light
        }
    }

    if (chosenLightType == POINT_LIGHT || chosenLightType == SPOT_LIGHT) {
        Lo += sumG;
        return Lo;
    }

    // ========== BSDF-Sampled Ray ==========
    // wiX = wi sampled w/r/t bsdf
    // sumF: 1/nSum f(x) * g(x) * w(x) / p(x)
    vec3 wiX;
    vec3 sumF = vec3(0.f);
    float pdfBsdfBsdf;                                                          // The PDF of your BSDF-sampled ray with respect to your intersection's BSDF = pdf_bsdf(wi_bsdf)
    vec2 xi = vec2(rng(), rng());
    int sampledType;
    vec3 bsdfBsdf = Sample_f(isect, woW, xi, wiX, pdfBsdfBsdf, sampledType);    // bsdf(p, wo, wi_bsdf)
    if (pdfBsdfBsdf > 0.f) {
        float pdfLiBsdf = Pdf_Li(view_point, isect.nor, wiX, chosenLightIdx);   // The PDF of your BSDF-sampled ray with respect to the light you chose with your light source sampling = pdf_Li(wi_bsdf)
        float wx = PowerHeuristic(1, pdfBsdfBsdf, 1, pdfLiBsdf);              // power: w(x) = nf * pdf_bsdf(wi_bsdf) / nf * pdf_bsdf(wi_bsdf) + ng * pdf_light(wi_bsdf)

        // Trace ray from BSDF sample
        Ray bsdfRay = SpawnRay(view_point, wiX);
        Intersection lightIsect = sceneIntersect(bsdfRay);
        // Accumulate light only if it hits the same light source
        if (lightIsect.obj_ID == chosenLightID) {
            vec3 Li_bsdf = lightIsect.Le;
            sumF = (bsdfBsdf * Li_bsdf * wx) / pdfBsdfBsdf;                     // sumF: bsdf(p, wo, wi_bsdf) * Li(p, wi_bsdf) * w(x) / pdf_bsdf
        }
    }

    if (any(isnan(Lo))) {
        Lo = vec3(0, 0, 0); // Set invalid colors to black
    }
    // average them together to produce an overall sample color
    Lo += (sumG + sumF);
    return Lo;
}

vec3 Li_Full(Ray ray) {
    vec3 Lo = vec3(0.f);
    vec3 throughput = vec3(1.f);
    bool prev_was_specular = false;
    vec3 woW = -ray.direction;

    for (int i = 0; i < MAX_DEPTH; ++i) {

        Intersection isect = sceneIntersect(ray);
        if (isect.t == INFINITY) {
          vec2 uv = sampleSphericalMap(normalize((ray.origin + isect.t * ray.direction)));
          Lo += texture(u_EnvironmentMap, uv).rgb * throughput;
          return Lo;
        }

        if (any(greaterThan(isect.Le, vec3(0.f)))) {
            if (i == 0 || prev_was_specular) {
                return isect.Le * throughput;
            } else {
                return Lo;
            }
        }

        // If we hit a specular object then MIS is pointless
        // Light-sampling produces no Lo, and BSD sampling always
        // produces the same ray, so wi_global_illum serves as the
        // direct illum & indirect illum ray for our specular surface
        if (isect.material.type != SPEC_REFL &&
            isect.material.type != SPEC_TRANS &&
            isect.material.type != SPEC_GLASS) {
            prev_was_specular = false;
            // Now that we have our isect on a non-light surface,
            // want to determine the direct illum on that point
            vec3 directLight = isectMIS(isect, woW, ray); // Light leaving the surface along
                                                    //wo that the surface received DIRECTLY from a light
            Lo += directLight * throughput;
        } else {
          prev_was_specular = true;
        }

        //  wi_global_illum will be used to determine the direction of ray bounce
        // and global illumination computation
        vec3 wi_global_illum;
        float pdf_gi;
        vec2 xi = vec2(rng(), rng());
        int sampledType;
        vec3 f = Sample_f(isect, woW, xi, wi_global_illum, pdf_gi, sampledType);

        // compounds the inherent material colors of all surfaces this ray
        // has bounced from so far, so that when we incorporate the lighting
        // this particular ray bounce receives directly, it is attenuated by
        // all of the surfaces our ray has previously hit
        if (pdf_gi > 0.f) {
          throughput *= f * AbsDot(wi_global_illum, isect.nor) / pdf_gi;
        }
        if (any(isnan(Lo))) {
            Lo = vec3(0, 0, 0); // Set invalid colors to black
        }
        ray = SpawnRay((ray.origin + isect.t * ray.direction), wi_global_illum);
    }
  return Lo;
}


void main()
{
    seed = uvec2(u_Iterations, u_Iterations + 1) * uvec2(gl_FragCoord.xy);

    Ray ray = rayCast();
    vec3 thisIterationColor;

    //Change #if to 1 for the Li_method you wish to run
#if 0
    thisIterationColor = Li_Naive(ray);
#endif
#if 0
    thisIterationColor = Li_Direct_Simple(ray);
#endif
#if 0
    thisIterationColor = Li_DirectMIS(ray);
#endif
#if 1
    thisIterationColor = Li_Full(ray);
#endif

    // when u_Iterations == 1, out_Col = thisIterationColor
    // when u_Iterations == 2, out_Col = (previous IterationColor + thisIterationColor) / n=2
    // when u_Iterations == n samples, out_Col = ((prevIterationColor * n-1) + thisIterationColor) / n
    // out_Col = mix(thisIter, prevIter, 1.0 / u_Iterations)

    // Set out_Col to the weighted sum of thisIterationColor
    // and all previous iterations' color values.
    // Refer to pathtracer.defines.glsl for what variables you may use
    // to acquire the needed values.

    vec3 prevIterationColor = texture(u_AccumImg, fs_UV).rgb;           // AccumImg: A texture storing the accumulation
                                                                        // of all previous iterations' color values
    float iterations = 1.f / float(u_Iterations);                       // Weight for current sample
    out_Col = vec4(mix(prevIterationColor, thisIterationColor, iterations), 1.f);

}
 

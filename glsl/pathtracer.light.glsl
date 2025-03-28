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

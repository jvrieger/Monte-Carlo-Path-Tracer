
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
            if (i == 0 || prev_was_specular) {
                vec2 uv = sampleSphericalMap(ray.direction);
                Lo += texture(u_EnvironmentMap, uv).rgb * throughput;
                return Lo;
            } else {
                return Lo;
            }
          
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

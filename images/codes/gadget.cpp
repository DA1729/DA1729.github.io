#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;
using int64 = long long;

// ===================================================================
// 1. UTILITY AND POLYNOMIAL ARITHMETIC FUNCTIONS
// (Based on standard LWE implementations)
// ===================================================================

// Positive modulo operation
int64 modq(int64 x, int64 q) {
    int64 r = x % q;
    return r < 0 ? r + q : r;
}

// Centered representation in (-q/2, q/2]
int64 center_rep(int64 x, int64 q) {
    int64 v = modq(x, q);
    return v > q / 2 ? v - q : v;
}

using poly = vector<int64>;

// c = a + b mod q
poly poly_add(const poly &a, const poly &b, int64 q) {
    size_t n = a.size();
    poly r(n);
    for (size_t i = 0; i < n; i++) r[i] = modq(a[i] + b[i], q);
    return r;
}

// c = a * s mod q (scalar multiplication)
poly poly_scalar_mul(const poly &a, int64 scalar, int64 q) {
    size_t n = a.size();
    poly r(n);
    for (size_t i = 0; i < n; i++) {
        __int128 tmp = (__int128)a[i] * scalar;
        r[i] = modq((int64)(tmp % q), q);
    }
    return r;
}

// c = a * b in R_q = Z_q[x]/(x^n+1)
poly poly_mul_negacyclic(const poly &a, const poly &b, int64 q) {
    size_t n = a.size();
    poly res(n, 0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            __int128 prod = (__int128)a[i] * b[j];
            if (i + j < n) {
                res[i + j] = modq(res[i + j] + (int64)(prod % q), q);
            } else {
                res[i + j - n] = modq(res[i + j - n] - (int64)(prod % q), q);
            }
        }
    }
    return res;
}

// Global random number generator
mt19937_64 rng((uint64_t)chrono::high_resolution_clock::now().time_since_epoch().count());


// ===================================================================
// 2. CRYPTOSYSTEM STRUCTURES AND FUNCTIONS
// ===================================================================

struct GLWE_ciphertext {
    vector<poly> D;
    poly B;
};

using GLev_ciphertext = vector<GLWE_ciphertext>;

// Encrypts a message M under the GLWE scheme
GLWE_ciphertext glwe_encrypt(const poly &M_scaled, const vector<poly> &PK1, const poly &PK0, int64 q, int64 noise_bound) {
    size_t n = PK0.size();
    int k = PK1.size();
    uniform_int_distribution<int> ud_bin(0, 1);
    uniform_int_distribution<int64> ud_noise(-noise_bound, noise_bound);

    // Sample small polynomials U, E1, E2
    poly U(n);
    for (auto &coeff : U) coeff = ud_bin(rng);

    poly E1(n);
    for (auto &coeff : E1) coeff = modq(ud_noise(rng), q);

    vector<poly> E2(k, poly(n));
    for (int i = 0; i < k; ++i) {
        for (auto &coeff : E2[i]) coeff = modq(ud_noise(rng), q);
    }

    // Compute ciphertext parts
    GLWE_ciphertext C;
    C.B = poly_mul_negacyclic(PK0, U, q);
    C.B = poly_add(C.B, M_scaled, q);
    C.B = poly_add(C.B, E1, q);

    C.D.resize(k);
    for (int i = 0; i < k; ++i) {
        C.D[i] = poly_mul_negacyclic(PK1[i], U, q);
        C.D[i] = poly_add(C.D[i], E2[i], q);
    }

    return C;
}


// Encrypts a message M into a GLev ciphertext
GLev_ciphertext glev_encrypt(const poly &M, const vector<poly> &PK1, const poly &PK0, int64 q, int64 t, int64 beta, int l, int64 noise_bound) {
    GLev_ciphertext C_glev;
    for (int i = 0; i < l; ++i) {
        int64_t g = 1;
        for(int j = 0; j < i + 1; ++j) g *= beta;
        int64_t delta_i = q / g;

        poly M_scaled = poly_scalar_mul(M, delta_i, q);
        C_glev.push_back(glwe_encrypt(M_scaled, PK1, PK0, q, noise_bound));
    }
    return C_glev;
}


// Decrypts a GLWE ciphertext
poly glwe_decrypt(const GLWE_ciphertext &C, const vector<poly> &S, int64 q, int64 t, int64 delta) {
    size_t n = C.B.size();
    int k = S.size();
    poly D_times_S(n, 0);
    for (int i = 0; i < k; ++i) {
        poly prod = poly_mul_negacyclic(C.D[i], S[i], q);
        D_times_S = poly_add(D_times_S, prod, q);
    }

    poly temp(n);
    for(size_t i=0; i<n; ++i) temp[i] = modq(C.B[i] - D_times_S[i], q);

    poly M_rec(n);
    for (size_t i = 0; i < n; i++) {
        int64 centered = center_rep(temp[i], q);
        int64 rounded = round((double)centered / delta);
        M_rec[i] = modq(rounded, t);
    }
    return M_rec;
}


// ===================================================================
// 3. GADGET DECOMPOSITION AND EXTERNAL PRODUCT
// ===================================================================

// Decomposes a polynomial 'p' into 'l' polynomials based on base 'beta'
vector<poly> gadget_decompose(const poly &p, int64_t q, int64_t beta, int l) {
    size_t n = p.size();
    vector<poly> decomp(l, poly(n, 0));
    poly current_rem = p;

    for (int i = 0; i < l; ++i) {
        int64_t g = 1;
        for(int j = 0; j < i + 1; ++j) g *= beta;
        g = q / g;

        if (g == 0) continue;

        for (size_t j = 0; j < n; ++j) {
            int64_t centered_rem = center_rep(current_rem[j], q);
            int64_t coeff = round((double)centered_rem / g);
            decomp[i][j] = coeff;
            __int128 rem_update = (__int128)coeff * g;
            current_rem[j] = modq(current_rem[j] - (int64_t)rem_update, q);
        }
    }
    return decomp;
}

// Performs the external product: decomp(phi) . C_glev
GLWE_ciphertext external_product(const vector<poly> &decomp_phi, const GLev_ciphertext &c_glev, int64 q) {
    size_t n = c_glev[0].B.size();
    int k = c_glev[0].D.size();
    int l = c_glev.size();

    GLWE_ciphertext result;
    result.B = poly(n, 0);
    result.D.assign(k, poly(n, 0));

    for (int i = 0; i < l; ++i) {
        // --- FIX: Use full polynomial multiplication instead of scalar ---
        poly b_scaled = poly_mul_negacyclic(c_glev[i].B, decomp_phi[i], q);
        
        vector<poly> d_scaled(k);
        for(int j=0; j<k; ++j) {
            d_scaled[j] = poly_mul_negacyclic(c_glev[i].D[j], decomp_phi[i], q);
        }
        // --- END FIX ---

        result.B = poly_add(result.B, b_scaled, q);
        for (int j = 0; j < k; ++j) {
            result.D[j] = poly_add(result.D[j], d_scaled[j], q);
        }
    }
    return result;
}

// Helper to calculate noise magnitude
int64 calculate_noise(const poly& p, const poly& expected, int64 q) {
    int64 max_noise = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        int64 noise = llabs(center_rep(p[i] - expected[i], q));
        if (noise > max_noise) max_noise = noise;
    }
    return max_noise;
}

// ===================================================================
// 4. MAIN DEMONSTRATION
//
//
// ===================================================================
// 4. MAIN DEMONSTRATION (CORRECTED)
// ===================================================================
int main() {
    // --- Parameters ---
    size_t n = 1024;
    int64_t q = 1LL << 32;
    int64_t t = 256;
    int k = 2;
    int64_t noise_bound = 8;
    int64_t delta = q / t;
    
    // Gadget parameters
    int64_t beta = 1024;
    int l = 3;

    int64_t M_val = 5;
    int64_t Phi_val = 12;

    cout << "------ Parameters ------" << endl;
    cout << "n: " << n << ", q: 2^32, t: " << t << ", k: " << k << endl;
    cout << "beta: " << beta << ", l: " << l << endl;
    cout << "Original Message M(x) = " << M_val << endl;
    cout << "Plaintext Multiplier Φ(x) = " << Phi_val << endl;
    cout << "Expected Result (M * Φ) = " << modq(M_val * Phi_val, t) << endl << endl;

    // --- Key Generation ---
    vector<poly> S(k, poly(n));
    uniform_int_distribution<int> ud_bin(0, 1);
    for (int i = 0; i < k; ++i) {
        for (auto &coeff : S[i]) coeff = ud_bin(rng);
    }

    vector<poly> A(k, poly(n));
    uniform_int_distribution<int64> ud_q(0, q - 1);
    for (int i = 0; i < k; ++i) {
        for (auto &coeff : A[i]) coeff = ud_q(rng);
    }
    
    poly E(n);
    uniform_int_distribution<int64> ud_noise(-noise_bound, noise_bound);
    for (auto &coeff : E) coeff = modq(ud_noise(rng), q);

    poly AS(n, 0);
    for (int i = 0; i < k; ++i) {
        AS = poly_add(AS, poly_mul_negacyclic(A[i], S[i], q), q);
    }
    poly P0 = poly_add(AS, E, q);
    vector<poly> P1 = A;

    // --- Operations ---
    cout << "------ Operations ------" << endl;
    poly M(n, 0); M[0] = M_val;
    poly Phi(n, 0); Phi[0] = Phi_val;
    
    cout << "Encrypting M=" << M_val << " into a GLev ciphertext..." << endl;
    GLev_ciphertext c_glev = glev_encrypt(M, P1, P0, q, t, beta, l, noise_bound);
    
    cout << "Decomposing Φ=" << Phi_val << " into (Φ_1, Φ_2, ...)..." << endl;
    vector<poly> decomp_phi = gadget_decompose(Phi, q, beta, l);
    
    cout << "Performing homomorphic external product..." << endl;
    GLWE_ciphertext c_final = external_product(decomp_phi, c_glev, q);
    
    // --- CORRECTED Decryption & Analysis for Gadget Method ---
    cout << "Decryption of the final ciphertext..." << endl;
    // 1. Get the phase: B' - D'S
    poly D_S_final(n, 0);
    for(int i=0; i<k; ++i) D_S_final = poly_add(D_S_final, poly_mul_negacyclic(c_final.D[i], S[i], q), q);
    poly phase_final(n);
    for(size_t i=0; i<n; ++i) phase_final[i] = modq(c_final.B[i] - D_S_final[i], q);

    // 2. To decrypt, we round the phase and reduce mod t (NO delta division)
    poly M_rec(n);
    for(size_t i=0; i<n; ++i) {
        int64 centered = center_rep(phase_final[i], q);
        M_rec[i] = modq(centered, t);
    }
    
    // --- Noise Analysis ---
    cout << "------ Noise Analysis ------" << endl;
    cout << "Correctness requires noise < q/(2t) = " << q / (2 * t) << endl << endl;

    // Corrected noise calculation for gadget method
    poly expected_phase_gadget = poly(n, 0);
    expected_phase_gadget[0] = M_val * Phi_val; // The expected phase is just M*Phi
    cout << "Noise from Gadget Method: " << calculate_noise(phase_final, expected_phase_gadget, q) << endl;

    // Noise from Naive Method (This part was correct for what it calculated)
    GLWE_ciphertext c_single = glwe_encrypt(poly_scalar_mul(M, delta, q), P1, P0, q, noise_bound);
    GLWE_ciphertext c_naive;
    c_naive.B = poly_scalar_mul(c_single.B, Phi_val, q);
    c_naive.D.resize(k);
    for(int i=0; i<k; ++i) c_naive.D[i] = poly_scalar_mul(c_single.D[i], Phi_val, q);
    // The naive result encrypts delta*M*Phi, so we decrypt it with delta
    poly M_rec_naive = glwe_decrypt(c_naive, S, q, t, delta);
    
    poly D_S_naive(n, 0);
    for(int i=0; i<k; ++i) D_S_naive = poly_add(D_S_naive, poly_mul_negacyclic(c_naive.D[i], S[i], q), q);
    poly phase_naive(n);
    for(size_t i=0; i<n; ++i) phase_naive[i] = modq(c_naive.B[i] - D_S_naive[i], q);
    
    poly expected_phase_naive(n,0);
    expected_phase_naive[0] = modq(M_val * Phi_val, t);
    expected_phase_naive = poly_scalar_mul(expected_phase_naive, delta, q);

    cout << "Noise from Naive Method: " << calculate_noise(phase_naive, expected_phase_naive, q) << endl << endl;
    
    int64 gadget_noise = calculate_noise(phase_final, expected_phase_gadget, q);
    int64 naive_noise = calculate_noise(phase_naive, expected_phase_naive, q);
    if (gadget_noise > 0) {
        cout << "The gadget-based multiplication resulted in noise ~" << naive_noise / gadget_noise << "x smaller than the naive approach!" << endl;
    }

    return 0;
}


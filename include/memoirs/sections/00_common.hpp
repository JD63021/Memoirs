#pragma once

// Auto-split from frozen patch014 one-shot source.
// This header is intentionally included in order by apps/poisson_mms.cpp.

// ============================================================================
// SECTION 0: Includes / precision / tiny utilities
// ============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#if defined(MEMOIRS_USE_HYPRE)
#include <mpi.h>
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#endif


#if defined(MEMOIRS_PRECISION_single)
using Real = float;
static constexpr const char* kPrecisionName = "single";
#else
using Real = double;
static constexpr const char* kPrecisionName = "double";
#endif

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

static Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static Vec3 operator*(double s, const Vec3& a) {
    return {s*a.x, s*a.y, s*a.z};
}

static Vec3 operator*(const Vec3& a, double s) {
    return {s*a.x, s*a.y, s*a.z};
}

static Vec3 cross3(const Vec3& a, const Vec3& b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}

static double dot3(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static double tet_signed_volume6(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    return dot3(b-a, cross3(c-a, d-a));
}

static std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) e--;
    return s.substr(b, e-b);
}

static bool starts_with(const std::string& s, const std::string& p) {
    return s.rfind(p, 0) == 0;
}

static std::string strip_comment(const std::string& line) {
    auto pos = line.find("//");
    if (pos == std::string::npos) return line;
    return line.substr(0, pos);
}

static std::vector<std::string> read_clean_lines(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    std::vector<std::string> lines;
    std::string line;
    bool in_block_comment = false;

    while (std::getline(f, line)) {
        std::string out;
        for (size_t i = 0; i < line.size();) {
            if (!in_block_comment && i + 1 < line.size() && line[i] == '/' && line[i+1] == '*') {
                in_block_comment = true;
                i += 2;
            } else if (in_block_comment && i + 1 < line.size() && line[i] == '*' && line[i+1] == '/') {
                in_block_comment = false;
                i += 2;
            } else if (!in_block_comment) {
                out.push_back(line[i++]);
            } else {
                i++;
            }
        }
        out = strip_comment(out);
        out = trim(out);
        if (!out.empty()) lines.push_back(out);
    }
    return lines;
}

static bool is_integer_line(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtol(s.c_str(), &end, 10);
    return end && *end == '\0';
}

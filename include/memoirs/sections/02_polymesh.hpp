#pragma once

// Auto-split from frozen patch014 one-shot source.
// This header is intentionally included in order by apps/poisson_mms.cpp.

// ============================================================================
// SECTION 2: OpenFOAM polyMesh reader
// ============================================================================

struct Face {
    std::vector<int> verts;
};

struct Patch {
    std::string name;
    std::string type;
    int nFaces = 0;
    int startFace = 0;
};

struct Cell {
    std::vector<int> faces;
    std::vector<int> verts;
};

struct PolyMesh {
    std::vector<Vec3> points;
    std::vector<Face> faces;
    std::vector<int> owner;
    std::vector<int> neighbour;
    std::vector<Patch> patches;
    std::vector<Cell> cells;
};

static std::vector<Vec3> read_points(const std::string& path) {
    auto lines = read_clean_lines(path);
    size_t i = 0;
    while (i < lines.size() && !is_integer_line(lines[i])) i++;
    if (i == lines.size()) throw std::runtime_error("Could not find point count in " + path);
    int n = std::stoi(lines[i++]);

    while (i < lines.size() && lines[i] != "(") i++;
    if (i == lines.size()) throw std::runtime_error("Could not find point list open paren in " + path);
    i++;

    std::vector<Vec3> pts;
    pts.reserve(n);

    for (; i < lines.size() && static_cast<int>(pts.size()) < n; ++i) {
        std::string s = lines[i];
        if (s.empty() || s == ")") continue;
        auto l = s.find('(');
        auto r = s.find(')');
        if (l == std::string::npos || r == std::string::npos || r <= l) continue;
        std::string inside = s.substr(l+1, r-l-1);
        std::istringstream iss(inside);
        Vec3 p;
        iss >> p.x >> p.y >> p.z;
        if (!iss) throw std::runtime_error("Bad point line: " + s);
        pts.push_back(p);
    }

    if (static_cast<int>(pts.size()) != n) {
        throw std::runtime_error("Point count mismatch in " + path);
    }
    return pts;
}

static std::vector<Face> read_faces(const std::string& path) {
    auto lines = read_clean_lines(path);
    size_t i = 0;
    while (i < lines.size() && !is_integer_line(lines[i])) i++;
    if (i == lines.size()) throw std::runtime_error("Could not find face count in " + path);
    int n = std::stoi(lines[i++]);

    while (i < lines.size() && lines[i] != "(") i++;
    if (i == lines.size()) throw std::runtime_error("Could not find face list open paren in " + path);
    i++;

    std::vector<Face> faces;
    faces.reserve(n);

    for (; i < lines.size() && static_cast<int>(faces.size()) < n; ++i) {
        std::string s = lines[i];
        if (s.empty() || s == ")") continue;

        auto l = s.find('(');
        auto r = s.find(')');
        if (l == std::string::npos || r == std::string::npos || r <= l) continue;

        int cnt = std::stoi(s.substr(0, l));
        std::string inside = s.substr(l+1, r-l-1);
        std::istringstream iss(inside);
        Face f;
        f.verts.resize(cnt);
        for (int k = 0; k < cnt; ++k) iss >> f.verts[k];
        if (!iss && cnt > 0) throw std::runtime_error("Bad face line: " + s);
        faces.push_back(std::move(f));
    }

    if (static_cast<int>(faces.size()) != n) {
        throw std::runtime_error("Face count mismatch in " + path);
    }
    return faces;
}

static std::vector<int> read_label_list(const std::string& path) {
    auto lines = read_clean_lines(path);
    size_t i = 0;
    while (i < lines.size() && !is_integer_line(lines[i])) i++;
    if (i == lines.size()) throw std::runtime_error("Could not find label count in " + path);
    int n = std::stoi(lines[i++]);

    while (i < lines.size() && lines[i] != "(") i++;
    if (i == lines.size()) throw std::runtime_error("Could not find label list open paren in " + path);
    i++;

    std::vector<int> vals;
    vals.reserve(n);

    for (; i < lines.size() && static_cast<int>(vals.size()) < n; ++i) {
        std::string s = lines[i];
        if (s.empty() || s == ")") continue;
        vals.push_back(std::stoi(s));
    }

    if (static_cast<int>(vals.size()) != n) {
        throw std::runtime_error("Label count mismatch in " + path);
    }
    return vals;
}

static std::vector<Patch> read_boundary(const std::string& path) {
    auto lines = read_clean_lines(path);
    size_t i = 0;
    while (i < lines.size() && !is_integer_line(lines[i])) i++;
    if (i == lines.size()) throw std::runtime_error("Could not find patch count in " + path);
    int n = std::stoi(lines[i++]);

    while (i < lines.size() && lines[i] != "(") i++;
    if (i == lines.size()) throw std::runtime_error("Could not find boundary list open paren in " + path);
    i++;

    std::vector<Patch> patches;
    patches.reserve(n);

    while (i < lines.size() && static_cast<int>(patches.size()) < n) {
        std::string name = lines[i++];
        if (name == ")" || name == "{" || name == "}") continue;

        while (i < lines.size() && lines[i] != "{") i++;
        if (i == lines.size()) throw std::runtime_error("Bad boundary patch block for " + name);
        i++;

        Patch p;
        p.name = name;

        while (i < lines.size() && lines[i] != "}") {
            std::string s = lines[i++];
            if (starts_with(s, "type")) {
                std::istringstream iss(s);
                std::string key, val;
                iss >> key >> val;
                if (!val.empty() && val.back() == ';') val.pop_back();
                p.type = val;
            } else if (starts_with(s, "nFaces")) {
                std::istringstream iss(s);
                std::string key;
                iss >> key >> p.nFaces;
            } else if (starts_with(s, "startFace")) {
                std::istringstream iss(s);
                std::string key;
                iss >> key >> p.startFace;
            }
        }
        if (i < lines.size() && lines[i] == "}") i++;
        patches.push_back(p);
    }

    if (static_cast<int>(patches.size()) != n) {
        throw std::runtime_error("Patch count mismatch in " + path);
    }

    return patches;
}

static PolyMesh read_polymesh(const std::string& dir) {
    PolyMesh m;
    m.points = read_points(dir + "/points");
    m.faces = read_faces(dir + "/faces");
    m.owner = read_label_list(dir + "/owner");
    m.neighbour = read_label_list(dir + "/neighbour");
    m.patches = read_boundary(dir + "/boundary");

    if (m.owner.size() != m.faces.size()) {
        throw std::runtime_error("owner size must equal faces size");
    }
    if (m.neighbour.size() > m.owner.size()) {
        throw std::runtime_error("neighbour size cannot exceed owner size");
    }

    int nCells = 0;
    for (int o : m.owner) nCells = std::max(nCells, o + 1);
    for (int nb : m.neighbour) nCells = std::max(nCells, nb + 1);

    m.cells.resize(nCells);

    for (int f = 0; f < static_cast<int>(m.faces.size()); ++f) {
        int c = m.owner[f];
        if (c < 0 || c >= nCells) throw std::runtime_error("Bad owner index");
        m.cells[c].faces.push_back(f);
    }
    for (int f = 0; f < static_cast<int>(m.neighbour.size()); ++f) {
        int c = m.neighbour[f];
        if (c < 0 || c >= nCells) throw std::runtime_error("Bad neighbour index");
        m.cells[c].faces.push_back(f);
    }

    for (auto& c : m.cells) {
        std::set<int> vs;
        for (int fid : c.faces) {
            for (int v : m.faces[fid].verts) vs.insert(v);
        }
        c.verts.assign(vs.begin(), vs.end());
    }

    return m;
}

#ifndef POLYGON_HPP
#define POLYGON_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>      // For std::abs
#include <numeric>    // For std::accumulate (optional, can use loop)
#include <stdexcept>  // For std::invalid_argument

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/multi/geometries/multi_polygon.hpp> // For intersection result


// Define aliases for convenience with Boost.Geometry
namespace bg = boost::geometry;
// We'll use double for Boost.Geometry calculations for better precision,
// even if input is float.
using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostMultiPolygon = bg::model::multi_polygon<BoostPolygon>;

// --- Point Representation (using input type) ---
// Alias for clarity
using Vertex = std::tuple<float, float>;
using PolygonVertices = std::vector<Vertex>;

class Polygon {
public:
    // --- Constructors ---
    Polygon() = default; // Default constructor

    explicit Polygon(const PolygonVertices& vertices_tuple, std::string id = "")
        : vertices_original_(vertices_tuple), id_(std::move(id)) {
        if (vertices_tuple.size() < 3 && !vertices_tuple.empty()) { // Allow empty polygon construction
            // We could throw, or just let it be an invalid polygon for area/intersection
            std::cerr << "Warning: Polygon '" << id_ << "' created with fewer than 3 vertices. Area and intersection might be zero or invalid." << std::endl;
        }
        // Convert to Boost.Geometry polygon representation immediately
        // for internal use and validation.
        boost_poly_ = toBoostPolygon(vertices_original_);
    }

    // --- Getters ---
    const PolygonVertices& getVertices() const {
        return vertices_original_;
    }

    const std::string& getId() const {
        return id_;
    }

    // --- Core Functionality ---

    // 1. Calculate the area of this polygon
    double getArea() const {
        if (vertices_original_.size() < 3) {
            return 0.0;
        }
        // Option 1: Use Shoelace formula on original float vertices (can be slightly less precise)
        // return calculateShoelaceArea(vertices_original_);

        // Option 2: Use Boost.Geometry's area calculation on the (corrected) double precision polygon
        // This is generally preferred for consistency if Boost.Geometry is used for intersection.
        try {
            return bg::area(boost_poly_);
        } catch (const bg::exception& e) {
            std::cerr << "Error calculating area for polygon '" << id_ << "' with Boost.Geometry: " << e.what() << std::endl;
            return 0.0; // Or rethrow, or return NaN
        }
    }

    // 2. Calculate the area of intersection with another polygon
    double getIntersectionArea(const Polygon& other) const {
        if (this->vertices_original_.size() < 3 || other.vertices_original_.size() < 3) {
            return 0.0; // Not valid polygons to intersect
        }

        // The intersection can result in multiple disjoint polygons
        BoostMultiPolygon intersection_result;
        try {
            bg::intersection(this->boost_poly_, other.boost_poly_, intersection_result);
        } catch (const bg::exception& e) {
            std::cerr << "Boost.Geometry intersection error between polygon '" << this->id_
                        << "' and '" << other.id_ << "': " << e.what() << std::endl;
            return 0.0;
        } catch (const std::exception& e) {
            std::cerr << "Standard C++ error during intersection between polygon '" << this->id_
                        << "' and '" << other.id_ << "': " << e.what() << std::endl;
            return 0.0;
        }

        double total_intersection_area = 0.0;
        if (!intersection_result.empty()) {
            for (const auto& p : intersection_result) {
                try {
                    total_intersection_area += bg::area(p);
                } catch (const bg::exception& e) {
                        std::cerr << "Boost.Geometry area calculation error for an intersection part involving polygon '"
                                << this->id_ << "' and '" << other.id_ << "': " << e.what() << std::endl;
                }
            }
        }
        return total_intersection_area;
    }

    // --- Helper for debugging ---
    bool isValidBoostPolygon() const {
        std::string reason;
        bool valid = bg::is_valid(boost_poly_, reason);
        if (!valid) {
            std::cerr << "Polygon '" << id_ << "' is not valid according to Boost.Geometry: " << reason << std::endl;
            // std::cerr << "WKT: " << bg::wkt(boost_poly_) << std::endl; // Requires <boost/geometry/io/wkt/wkt.hpp>
        }
        return valid;
    }


private:
    PolygonVertices vertices_original_; // Store original vertices as provided
    BoostPolygon boost_poly_;       // Internal Boost.Geometry representation
    std::string id_;                    // Optional identifier

    // Helper function to convert your vertex format to Boost.Geometry polygon
    static BoostPolygon toBoostPolygon(const PolygonVertices& vertices_tuple) {
        BoostPolygon poly;
        if (vertices_tuple.empty()) {
            return poly; // Return empty polygon
        }

        for (const auto& tpl : vertices_tuple) {
            bg::append(poly.outer(), BoostPoint(static_cast<double>(std::get<0>(tpl)),
                                                static_cast<double>(std::get<1>(tpl))));
        }

        // Ensure the polygon is closed for Boost.Geometry
        // (first and last point are the same for the outer ring).
        // bg::correct handles this, but an explicit check can be illustrative.
        if (!vertices_tuple.empty()) {
            const auto& first_tpl = vertices_tuple.front();
            const auto& last_tpl = vertices_tuple.back();
                // Check if not already closed by tuple data
            if (std::get<0>(first_tpl) != std::get<0>(last_tpl) || std::get<1>(first_tpl) != std::get<1>(last_tpl)) {
                bg::append(poly.outer(), BoostPoint(static_cast<double>(std::get<0>(first_tpl)),
                                                    static_cast<double>(std::get<1>(first_tpl))));
            }
        }
        
        // Correct the polygon (e.g., ensures winding order, closes it if not already, fixes minor issues).
        // This is crucial for robust operation of many Boost.Geometry algorithms.
        try {
            bg::correct(poly);
        } catch (const bg::exception& e) {
            std::cerr << "Warning: Boost.Geometry bg::correct failed: " << e.what() << std::endl;
            // Polygon might be severely invalid (e.g., self-intersecting in complex ways)
            // Further operations might fail or give unexpected results.
        }
        return poly;
    }

    // Shoelace formula (alternative for area, primarily for float precision if needed, or as a fallback)
    static double calculateShoelaceArea(const PolygonVertices& vertices) {
        if (vertices.size() < 3) {
            return 0.0;
        }

        double area_sum = 0.0;
        int n = vertices.size();

        for (int i = 0; i < n; ++i) {
            // Using double for intermediate calculations to maintain precision from float
            double x1 = static_cast<double>(std::get<0>(vertices[i]));
            double y1 = static_cast<double>(std::get<1>(vertices[i]));
            double x2 = static_cast<double>(std::get<0>(vertices[(i + 1) % n]));
            double y2 = static_cast<double>(std::get<1>(vertices[(i + 1) % n]));

            area_sum += (x1 * y2 - x2 * y1);
        }
        return std::abs(area_sum) / 2.0;
    }
};    

#endif // POLYGON_HPP
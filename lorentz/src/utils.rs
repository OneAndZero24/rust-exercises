use std::ops::{Add, Sub, Mul, Div};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

// Point + Point
impl Add for Point3D {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

// Point - Point
impl Sub for Point3D {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

// Point * scalar
impl Mul<f64> for Point3D {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

// scalar * Point
impl Mul<Point3D> for f64 {
    type Output = Point3D;

    fn mul(self, point: Point3D) -> Point3D {
        Point3D {
            x: self * point.x,
            y: self * point.y,
            z: self * point.z,
        }
    }
}

// Point / scalar
impl Div<f64> for Point3D {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

pub trait System {
    fn derivative(&self, point: Point3D) -> Point3D;
}
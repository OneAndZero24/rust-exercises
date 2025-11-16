use crate::utils::{Point3D, System};

#[derive(Debug, Clone, Copy)]
pub struct LorentzSystem {
    sigma: f64,
    r: f64,
    b: f64,
}

impl LorentzSystem {
    pub fn new(sigma: f64, r: f64, b: f64) -> Self {
        Self { sigma, r, b }
    }
}

impl Default for LorentzSystem {
    fn default() -> Self {
        Self { sigma: 10.0, r: 28.0, b: 8.0 / 3.0 }
    }
}

impl System for LorentzSystem {
    fn derivative(&self, point: Point3D) -> Point3D {
        Point3D {
            x: self.sigma * (point.y - point.x),
            y: point.x * (self.r - point.z) - point.y,
            z: point.x * point.y - self.b * point.z,
        }
    }
}
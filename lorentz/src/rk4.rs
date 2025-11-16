use crate::utils::{Point3D, System};

pub struct RK4Integrator<S: System> {
    dt: f64,
    system: S,
}

impl<S: System> RK4Integrator<S> {
    pub fn new(dt: f64, system: S) -> Self {
        Self { dt, system }
    }

    pub fn forward(&self, point: Point3D) -> Point3D {
        let h = self.dt;
        let k1 = self.system.derivative(point);
        let k2 = self.system.derivative(point+0.5*k1*h);
        let k3 = self.system.derivative(point+0.5*k2*h);
        let k4 = self.system.derivative(point+k3*h);

        point + (k1 + 2.0*k2 + 2.0*k3 + k4)*(h/6.0)
    }
}

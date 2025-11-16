mod utils;
mod lorentz;
mod rk4;

use clap::Parser;
use lorentz::LorentzSystem;
use rk4::RK4Integrator;
use utils::Point3D;
use plotters::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Starting x coordinate
    #[arg(short, long, default_value_t = 1.0)]
    x: f64,

    /// Starting y coordinate
    #[arg(short, long, default_value_t = 1.0)]
    y: f64,

    /// Starting z coordinate
    #[arg(short, long, default_value_t = 1.0)]
    z: f64,

    /// Number of steps to simulate
    #[arg(short, long, default_value_t = 10000)]
    steps: usize,

    /// Time step size
    #[arg(long, default_value_t = 0.01)]
    dt: f64,

    /// Lorenz sigma parameter
    #[arg(long, default_value_t = 10.0)]
    sigma: f64,

    /// Lorenz r parameter
    #[arg(long, default_value_t = 28.0)]
    r: f64,

    /// Lorenz b parameter
    #[arg(long, default_value_t = 8.0 / 3.0)]
    b: f64,

    /// Output file path
    #[arg(short, long, default_value = "lorentz.png")]
    output: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    let start = Point3D {
        x: args.x,
        y: args.y,
        z: args.z,
    };
    
    let system = LorentzSystem::new(args.sigma, args.r, args.b);
    let integrator = RK4Integrator::new(args.dt, system);
    
    // Generate trajectory
    let mut trajectory = Vec::with_capacity(args.steps);
    let mut current = start;
    trajectory.push(current);
    
    for _ in 0..args.steps {
        current = integrator.forward(current);
        trajectory.push(current);
    }
    
    // Plot the trajectory
    let root = BitMapBackend::new(&args.output, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let (x_min, x_max) = trajectory.iter().map(|p| p.x).fold((f64::INFINITY, f64::NEG_INFINITY), 
        |(min, max), x| (min.min(x), max.max(x)));
    let (y_min, y_max) = trajectory.iter().map(|p| p.y).fold((f64::INFINITY, f64::NEG_INFINITY), 
        |(min, max), y| (min.min(y), max.max(y)));
    let (z_min, z_max) = trajectory.iter().map(|p| p.z).fold((f64::INFINITY, f64::NEG_INFINITY), 
        |(min, max), z| (min.min(z), max.max(z)));
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Lorenz Attractor", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_3d(x_min..x_max, z_min..z_max, y_min..y_max)?;
    
    chart.configure_axes().draw()?;
    
    chart.draw_series(LineSeries::new(
        trajectory.iter().map(|p| (p.x, p.z, p.y)),
        &BLUE,
    ))?;
    
    root.present()?;
    println!("Plot saved to {}", args.output);
    
    Ok(())
}

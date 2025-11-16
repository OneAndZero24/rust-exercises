use cudarc::driver::*;
use rand::Rng;
use image::{RgbImage, Rgb};

fn compile_ptx(src: &str) -> Result<cudarc::nvrtc::Ptx, Box<dyn std::error::Error>> {
    use cudarc::nvrtc::compile_ptx_with_opts;
    
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_52"),
        ..Default::default()
    };
    
    Ok(compile_ptx_with_opts(src, opts)?)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = match h {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing CUDA device...");
    
    // Initialize CUDA device
    let dev = CudaDevice::new(0)?;
    println!("Using device: {}", dev.name()?);
    
    // Load and compile the kernel
    println!("Compiling CUDA kernel...");
    let kernel_src = std::fs::read_to_string("cuda-jfa/src/kernel.cu")?;
    let ptx = compile_ptx(&kernel_src)?;
    dev.load_ptx(ptx, "jfa_module", &["jfa_step"])?;
    
    let n = 10;
    let size = 512;
    let total_pixels = size * size;
    
    let mut lookup = vec![0i32; n * 2]; // Flattened: [x0, x1, ..., y0, y1, ...]
    let mut seed_image = vec![-1i32; total_pixels]; // -1 means unassigned
    let mut rng = rand::thread_rng();
    
    let mut seeds = vec![(0i32, 0i32); n];

    for i in 0..n{
        let x = rng.gen_range(0..size);
        let y = rng.gen_range(0..size);
        let idx = y * size + x;
        seeds[i] = (x as i32, y as i32);
        seed_image[idx] = i as i32;
        lookup[i] = x as i32;
        lookup[i + n] = y as i32;
    }
    
    // Allocate device memory
    let mut image_a_dev = dev.htod_sync_copy(&seed_image)?;
    let mut image_b_dev = dev.htod_sync_copy(&seed_image)?;
    let lookup_dev = dev.htod_sync_copy(&lookup)?;
    
    // Get kernel
    let jfa_func = dev.get_func("jfa_module", "jfa_step").unwrap();
    
    // Configure launch parameters
    let block_dim = (16, 16, 1);
    let grid_dim = (
        (size + block_dim.0 - 1) / block_dim.0,
        (size + block_dim.1 - 1) / block_dim.1,
        1
    );
    let cfg = LaunchConfig {
        grid_dim: (grid_dim.0 as u32, grid_dim.1 as u32, grid_dim.2 as u32),
        block_dim: (block_dim.0 as u32, block_dim.1 as u32, block_dim.2 as u32),
        shared_mem_bytes: 0,
    };
    
    println!("Launch config - Grid: {:?}, Block: {:?}", grid_dim, block_dim);
    
    let mut step_size = (size as f32).log2().ceil() as i32;
    step_size = 1 << (step_size - 1); // Nearest power of 2
    
    let mut ping = true;
    while step_size >= 1 {
            let (input, output) = if ping {
                (&image_a_dev, &mut image_b_dev)
            } else {
                (&image_b_dev, &mut image_a_dev)
            };

        unsafe {
            jfa_func.clone().launch(
                cfg,
                (input, output, &lookup_dev, size as i32, step_size)
            )?;
        }
        
        dev.synchronize()?;
        ping = !ping;
        step_size /= 2;
    }
    
    // Copy result back to host
    let result_buffer = if ping { &image_a_dev } else { &image_b_dev };
    let result: Vec<i32> = dev.dtoh_sync_copy(result_buffer)?;
    
    println!("\nJump Flood Algorithm completed!");
    
    println!("Generating output image...");
    let mut img = RgbImage::new(size as u32, size as u32);
    
    let colors: Vec<[u8; 3]> = (0..n-1).map(|i| {
        let hue = (i as f32 / n as f32) * 360.0;
        hsv_to_rgb(hue, 0.8, 0.9)
    }).collect();
    
    for y in 0..size {
        for x in 0..size {
            let idx = y * size + x;
            let seed_id = result[idx] as usize;
            let color = colors[seed_id % colors.len()];
            img.put_pixel(x as u32, y as u32, Rgb(color));
        }
    }

    for & (x, y) in seeds.iter() {
        // Draw a small circle at each seed point
        for dy in -2..=2 {
            for dx in -2..=2 {
                if dx * dx + dy * dy <= 4 { // Circle with radius ~2
                    let px = (x + dx).max(0).min(size as i32 - 1);
                    let py = (y + dy).max(0).min(size as i32 - 1);
                    img.put_pixel(px as u32, py as u32, Rgb([0, 0, 0]));
                }
            }
        }
    }
    
    img.save("output.png")?;
    println!("Saved output image to output.png");
    
    Ok(())
}
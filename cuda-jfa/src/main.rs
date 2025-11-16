use cudarc::driver::*;
use rand::Rng;
use image::{RgbImage, Rgb};

fn compile_ptx(src: &str) -> Result<cudarc::nvrtc::Ptx, Box<dyn std::error::Error>> {
    use cudarc::nvrtc::Ptx;
    
    Ok(Ptx::from_src(src))
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
    let kernel_src = std::fs::read_to_string("src/kernel.cu")?;
    let ptx = compile_ptx(&kernel_src)?;
    dev.load_ptx(ptx, "jfa_module", &["jfa_step"])?;
    
    let n = 10;
    let size = 512;
    let total_pixels = size * size;
    
    let mut lookup = vec![vec![0u32; n]; 2];
    let mut seed_image = vec![0u8; total_pixels];
    let mut rng = rand::thread_rng();
    
    for i in 0..n-1 {
        let x = rng.gen_range(0..size);
        let y = rng.gen_range(0..size);
        let idx = y * size + x;
        seed_image[idx] = (i) as u8;
        lookup[0][i] = x as u32;
        lookup[1][i] = y as u32;
    }
    
    // Allocate device memory
    let image_a_dev = dev.htod_sync_copy(&image_a)?;
    let image_b_dev = dev.htod_sync_copy(&image_b)?;
    let lookup_dev = dev.htod_sync_copy(&lookup)?;
    
    // Get kernel
    let jfa_func = dev.get_func("jfa_module", "jfa_step").unwrap();
    
    // Configure launch parameters
    let block_dim = (16, 16, 1);
    let grid_dim = (
        (width + block_dim.0 - 1) / block_dim.0,
        (height + block_dim.1 - 1) / block_dim.1,
        1
    );
    
    println!("Launch config - Grid: {:?}, Block: {:?}", grid_dim, block_dim);
    
    let mut step_size = (width as f32).log2().ceil() as i32;
    step_size = 1 << (step_size - 1); // Nearest power of 2
    
    let (input, output) = (&image_a_dev, &mut image_b_dev);

    while step_size >= 1 {
        unsafe {
            jfa_func.launch(
                cfg,
                (input.as_device_ptr(), output.as_device_ptr(), lookup.as_device_ptr(), size as i32, step_size)
            )?;
        }
        
        dev.synchronize()?;
        (input, output) = (output, input);
        step_size /= 2;
    }
    
    // Copy result back to host
    let result_buffer = if step_size%2 == 0 { &image_a_dev } else { &image_b_dev };
    let result: Vec<u8> = dev.dtoh_sync_copy(result_buffer)?;
    
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
    
    img.save("output.png")?;
    println!("Saved output image to output.png");
    
    Ok(())
}
use std::time::Instant;
use pixel_canvas::{Canvas, Color, input::MouseState};
use pixel_canvas::canvas::CanvasInfo;
use pixel_canvas::input::Event;
use pixel_canvas::input::glutin::event::{DeviceEvent, ElementState, KeyboardInput, VirtualKeyCode};
use rayon::prelude::*;

pub enum BoundaryConditions<T: Number>{
    Wrap,
    Fixed(T)
}

impl<T: Number> BoundaryConditions<T> {
    fn get(&self, grid: &Vec<Vec<T>>, i: usize, j: usize) -> T {
        match self {
            BoundaryConditions::Wrap => grid[(i + grid.len()) % grid.len()][(j + grid.len()) % grid.len()],
            BoundaryConditions::Fixed(num) => if i < 0 || i >= grid.len() || j < 0 || j >= grid.len() {
                *num
            } else {
                grid[i][j]
            }
        }
    }

    fn get_inc_dec(&self, grid: &Vec<Vec<T>>) -> (Box<dyn Fn(usize) -> usize + Send + Sync>, Box<dyn Fn(usize) -> usize + Send + Sync>) {
        let size = grid.len();

        match self {
            BoundaryConditions::Wrap => (
                Box::new(move |x| if x >= size - 1 { 0 } else { x + 1 }),
                Box::new(move |x| if x == 0 { size - 1 } else { x - 1 })
            ),
            BoundaryConditions::Fixed(_) => (
                Box::new(move |x| if x >= size - 1 { size - 1 } else { x + 1 }),
                Box::new(move |x| if x == 0 { x } else { x - 1 })
            )
        }
    }

    fn change_point(&self, grid: &Vec<Vec<T>>, point: (i32, i32)) -> (usize, usize) {
        match self {
            BoundaryConditions::Wrap => (
                (point.0 + grid.len() as i32) as usize % grid.len(),
                (point.1 + grid.len() as i32) as usize % grid.len(),
            ),
            BoundaryConditions::Fixed(_) => (
                point.0.clamp(0, grid.len() as i32 - 1) as usize,
                point.1.clamp(0, grid.len() as i32 - 1) as usize,
            )
        }
    }
}

pub struct WaveState<T: Number> {
    pub grid_size: usize,
    pub scale: usize,
    pub grid: Vec<Vec<T>>,
    pub prev_grid: Vec<Vec<T>>,
    pub mouse_state: MouseState,
    pub mouse_down: bool,
    pub click_value: T,
    pub dx: f64,
    pub dt: f64,
    pub v: f64,
    pub boundary_conditions: BoundaryConditions<T>,
    pub brush_size: usize,
    pub thing: Box<dyn Fn(f64, f64, f64) -> Option<T> + Send + Sync>,
    pub start: Instant
}

impl<T: Number> WaveState<T> {
    fn handle_input(info: &CanvasInfo, wave_state: &mut WaveState<T>, event: &Event<()>) -> bool {
        match event {
            Event::DeviceEvent { event: DeviceEvent::Button { state, .. }, ..} => {
                wave_state.mouse_down = *state == ElementState::Pressed;

                true
            },
            Event::DeviceEvent { event: DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(VirtualKeyCode::W),
                ..
            }), .. } => {
                wave_state.v += 0.01;
                println!("{:.3}", wave_state.v * (wave_state.dt * wave_state.dt) / (wave_state.dx * wave_state.dx));

                true
            },
            Event::DeviceEvent { event: DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(VirtualKeyCode::S),
                ..
            }), .. } => {
                wave_state.v += 0.01;
                println!("{:.3}", wave_state.v * (wave_state.dt * wave_state.dt) / (wave_state.dx * wave_state.dx));

                true
            },
            event => MouseState::handle_input(info, &mut wave_state.mouse_state, event)
        }
    }

    fn update(&mut self) {
        let grid = std::mem::replace(&mut self.grid, Vec::new());
        let (inc, dec) = self.boundary_conditions.get_inc_dec(&grid);

        self.grid = new_grid(grid.len(), |x, y| {
            if let Some(num) = (self.thing)(
                x as f64 / self.grid_size as f64,
                y as f64 / self.grid_size as f64,
                Instant::now().duration_since(self.start).as_secs_f64()
            ) {
                num
            } else {
                let dw2_dx2 = (grid[inc(x)][y] + grid[dec(x)][y] - grid[x][y] * 2.0);
                let dw2_dy2 = (grid[x][inc(y)] + grid[x][dec(y)] - grid[x][y] * 2.0);
                // let dw2_dxy2 = (grid[inc(x)][inc(y)] + grid[dec(x)][dec(y)] - grid[x][y] * 2.0) / 2.0f64.sqrt();
                // let dw2_dyx2 = (grid[dec(x)][inc(y)] + grid[inc(x)][dec(y)] - grid[x][y] * 2.0) / 2.0f64.sqrt();
                let dw_dt = grid[x][y] * 2.0 - self.prev_grid[x][y];

                dw_dt + (dw2_dx2 + dw2_dy2) * self.v * (self.dt * self.dt) / (self.dx * self.dx)
            }
        });

        self.prev_grid = grid;
    }
}

fn main() {
    let grid_size = 500;
    let scale = 1;
    
    let canvas = Canvas::new(scale * grid_size, scale * grid_size)
        .title("Wave Function")
        .show_ms(true)
        .state(WaveState {
            grid_size,
            scale,
            grid: new_grid(grid_size, |_, _| Complex(0.0, 0.0)),
            prev_grid: new_grid(grid_size, |_, _| Complex(0.0, 0.0)),
            mouse_state: MouseState::new(),
            mouse_down: false,
            click_value: Complex(1.0, 0.0),
            dt: 1.0 / 60.0,
            dx: 1.0 / (grid_size as f64),
            v: 0.005,
            boundary_conditions: BoundaryConditions::Fixed(Complex(0.0, 0.0)),
            brush_size: 8,
            thing: Box::new(|x, y, t| {
                // if (x - 0.5).powi(2) + (y - 0.5).powi(2) > 0.25f64.powi(2) {
                //     Some(0.0)
                // } else
                // if (x - 0.5).powi(2) + (y - 0.5).powi(2) < 0.05f64.powi(2) {
                //     Some((2.0 * t).cos() + (2.1 * t).cos())
                // } else {
                //    None
                // }
                // let w = 0.01;
                // let d = 0.005;
                //
                // if 0.8 - d < x && x < 0.8 &&
                //     !(0.4 - w < y && y < 0.4) &&
                //     !(0.6 < y && y < 0.6 + w)
                // {
                //     Some(Complex(0.0, 0.0))
                // } else
                if x > 0.99 {
                    let a = 1.0 * t;
                    Some(Complex(a.cos(), a.sin()))
                } else {
                    None
                }
                // None
            }),
            start: Instant::now()
        })
        .input(WaveState::handle_input);
    
    canvas.render(move |wave_state, image| {
        wave_state.update();
        // wave_state.update();
        // wave_state.update();
        // wave_state.update();

        let WaveState {
            grid,
            prev_grid,
            scale,
            mouse_down,
            boundary_conditions,
            mouse_state,
            click_value,
            brush_size,
            thing,
            start,
            ..
        } = wave_state;

        if *mouse_down {
            let (x, y) = (
                (mouse_state.x / (2 * *scale) as i32) as usize,
                (mouse_state.y / (2 * *scale) as i32 + grid_size as i32 / 2) as usize
            );

            let arr: Vec<(usize, usize)> = (-(*brush_size as i32)..*brush_size as i32).fold(Vec::new(), |mut v, i| {
                v.append(&mut (-(*brush_size as i32)..*brush_size as i32).fold(Vec::new(), |mut v, j| {
                    if ((i*i + j*j) as usize) < brush_size.pow(2) {
                        v.push(boundary_conditions.change_point(&grid, (
                            x as i32 + i,
                            y as i32 + j
                        )));
                    }

                    v
                }));

                v
            });
            println!("hi");

            for (i, j) in arr {
                grid[i][j] = *click_value * (Instant::now().duration_since(*start).as_secs_f64()*2.0).cos();
                prev_grid[i][j] = *click_value * (Instant::now().duration_since(*start).as_secs_f64()*2.0).cos();
            }

        }

        let (min, max) = grid
            .par_iter()
            .map(|row| row
                .par_iter()
                .map(|n| (n.get_magnitude(), n.get_magnitude()))
                .reduce(|| (100000.0, -100000.0), |(n1, n2), (n3, n4)| {
                    (n1.min(n3), n2.max(n4))
                })
            ).reduce(|| (100000.0, -100000.0), |(n1, n2), (n3, n4)| {
                (n1.min(n3), n2.max(n4))
            });

        println!("min: {:.5}, max: {:.5}", min, max);

        // let (min, max) = (-1.0, 1.0);

        // let diff = max - min;

        let width = image.width();
        for (y, row) in image.chunks_mut(width).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                *pixel = grid[x / *scale][y / *scale].get_color();
                pixel.b = if thing(
                    x as f64 / *scale as f64 / grid_size as f64,
                    y as f64 / *scale as f64 / grid_size as f64,
                    Instant::now().duration_since(*start).as_secs_f64()
                ).is_some() { 200 } else { pixel.b };
            }
        }
    });
}

fn new_grid<T: Number>(size: usize, func: impl Fn(usize, usize) -> T + Send + Sync) -> Vec<Vec<T>> {
    let x: Vec<usize> = (0..size).collect();

    x.par_iter()
        .map(|i| x.par_iter()
            .map(|j| func(*i, *j))
            .collect()
        ).collect()
}

pub struct Number2<T>(T);

pub trait Number:
    std::ops::Add<Output=Self> +
    std::ops::Sub<Output=Self> +
    std::ops::Mul<f64,Output=Self> +
    std::ops::Div<f64,Output=Self> + Copy + Send + Sync {
    fn get_magnitude(&self) -> f64;

    fn get_color(&self) -> Color {
        let x = self.get_magnitude();

        Color::rgb(
            (128.0 * x + 128.0) as u8,
            50,
            50
        )
    }
}

impl Number for f64 {
    fn get_magnitude(&self) -> f64 {
        *self
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        Complex(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Complex(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Complex(rhs * self.0, rhs * self.1)
    }
}

impl std::ops::Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Complex(self.0 / rhs, self.1 / rhs)
    }
}

impl Copy for Complex {

}

#[derive(Clone)]
struct Complex(f64, f64);

impl Number for Complex {
    fn get_magnitude(&self) -> f64 {
        // self.0.signum() * (self.0 * self.0 + self.1 * self.1).sqrt()
        self.0
    }

    fn get_color(&self) -> Color {
        let a = self.1.atan2(self.0);
        let m = 256.0 * (self.0 * self.0 + self.1 * self.1).sqrt();

        Color::rgb(
            (m * ((a).cos() + 1.0) / 2.0) as u8,
            (m * ((a + 3.14 * 2.0 / 3.0).cos() + 1.0) / 2.0) as u8,
            (m * ((a + 3.14 * 4.0 / 3.0).cos() + 1.0) / 2.0) as u8,
        )
    }
}
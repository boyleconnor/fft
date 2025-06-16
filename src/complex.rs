use std::ops::Sub;

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct Complex {
    pub real: f64,
    pub imaginary: f64
}

impl Complex {
    pub fn zero() -> Complex {
        Complex { real: 0.0, imaginary: 0.0 }
    }

    pub fn real(real: f64) -> Complex {
        Complex { real, imaginary: 0.0 }
    }

    /// Give the value of e^(ix)
    pub fn unity_root(x: f64) -> Complex {
        let (sin, cos) = f64::sin_cos(x);
        Complex { real: cos, imaginary: sin }
    }

    /// Give the value of e^(2πix/N), i.e. an x/n rotation in the complex plane, starting at 1
    pub fn w(x: usize, n: usize, sign: f64) -> Complex {
        Complex::unity_root(sign * 2.0 * std::f64::consts::PI * x as f64 / n as f64)
    }

    pub fn euclidean_distance(&self, other: Complex) -> f64 {
        (
            (self.imaginary - other.imaginary).powf(2.0) +
            (self.real - other.real).powf(2.0)
        ).powf(0.5)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            imaginary: self.imaginary + rhs.imaginary
        }
    }
}

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imaginary += rhs.imaginary;
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, rhs: Self) -> Self::Output {
        Complex {
            real: self.real - rhs.real,
            imaginary: self.imaginary - rhs.imaginary
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            real: (self.real * rhs.real) - (self.imaginary * rhs.imaginary),
            imaginary: (self.real * rhs.imaginary) + (self.imaginary * rhs.real)
        }
    }
}

impl std::ops::MulAssign for Complex {
    // FIXME: Reduce the allocations required for this
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

#[test]
fn test_complex_mult() {
    assert_eq!(Complex::zero(), Complex::zero());
    assert_eq!(Complex { real: 1.0, imaginary: 1.0 } * Complex { real: 1.0, imaginary: 1.0}, Complex { real: 0.0, imaginary: 2.0 });
    assert_eq!(Complex { real: 0.0, imaginary: 1.0 } * Complex { real: 1.0, imaginary: 1.0}, Complex { real: -1.0, imaginary: 1.0 });
    let (i, minus_one, one, minus_i) = (
        Complex { real: 0.0, imaginary: 1.0 },
        Complex { real: -1.0, imaginary: 0.0 },
        Complex { real: 1.0, imaginary: 0.0 },
        Complex { real: 0.0, imaginary: -1.0 }
    );

    assert!(Complex::unity_root(std::f64::consts::PI / 2.0).euclidean_distance(i) < 0.001);
    assert!(Complex::unity_root(std::f64::consts::PI).euclidean_distance(minus_one) < 0.001);
    assert!(Complex::unity_root(std::f64::consts::PI * 2.0).euclidean_distance(one) < 0.001);
    assert!(Complex::unity_root(std::f64::consts::PI * 3.0 / 2.0).euclidean_distance(minus_i) < 0.001);
}
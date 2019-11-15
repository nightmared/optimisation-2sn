use ndarray::{Array, Array1, Array2, ArrayBase};
use ndarray_linalg::Norm;

use ndarray::prelude::*;
use ndarray_linalg::Solve;

const EPSILON: f64 = 1e-12;

#[derive(Debug)]
enum ExitCondition {
    MaxIterationReached,
    NullGradient,
    FixedPoint
}

#[derive(Debug)]
struct AlgoParams {
    epsilon: f64,
    kMax: usize
}

fn newton(gradient: impl Fn(&Array1<f64>) -> Array1<f64>, hessienne: impl Fn(&Array1<f64>) -> Array2<f64>, x0: &Array1<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
    let mut xk = x0.clone();
    let mut dk = hessienne(&xk).solve_into(-gradient(&xk)).unwrap();

    let mut k = 0;
    loop {
        xk = xk+dk;
        k += 1;

        dk = hessienne(&xk).solve_into(-gradient(&xk)).unwrap();
        if k == params.kMax {
            return (xk, ExitCondition::MaxIterationReached);
        } else if gradient(&xk).norm()/(gradient(x0).norm()+1e-8) < params.epsilon {
            return (xk, ExitCondition::NullGradient);
        } else if dk.norm()/(xk.norm()+1e-8) < params.epsilon {
            return (xk, ExitCondition::FixedPoint);
        }

    }
}

fn pas_de_cauchy(gk: &Array1<f64>, hk: &Array2<f64>, delta_k: f64, epsilon: f64) -> Array1<f64> {
    if gk.norm() < epsilon {
        return Array1::zeros(gk.len());
    }

    let a = gk.t().dot(&hk.dot(gk))/2.;
    let b = -gk.norm().powi(2);
    let t_max = delta_k/gk.norm();

    if a <= 0. {
        -t_max*gk
    } else {
        -f64::min(-b/(2.*a), t_max)*gk
    }
}

fn f1(x1: f64, x2: f64, x3: f64) -> f64 {
    2.*(x1+x2+x3-3.).powi(2)+(x1-x2).powi(2)+(x2-x3).powi(2)
}

fn gradient_f1(x: &Array1<f64>) -> Array1<f64> {
    array![
        4.*(x[0]+x[1]+x[2]-3.) + 2.*(x[0]-x[1]),
        4.*(x[0]+x[1]+x[2]-3.) - 2.*(x[0]-x[1]) + 2.*(x[1]-x[2]),
        4.*(x[0]+x[1]+x[2]-3.) - 2.*(x[1]-x[2])
    ]
}

fn hessienne_f1(x: &Array1<f64>) -> Array2<f64> {
    array![
        [6., 2., 4.],
        [2., 8., 2.],
        [4., 2., 6.]
    ]
}

fn f2(x1: f64, x2: f64) -> f64 {
    100.*(x2-x1.powi(2)).powi(2)+(1.-x1).powi(2)
}

//fn f2_array(x: &Array1<f64>) -> f64 {
//    100.*(x[1]-x[0].powi(2)).powi(2)+(1.-x[0]).powi(2)
//}

fn gradient_f2(x: &Array1<f64>) -> Array1<f64> {
    array![
        -300.*x[0]*x[1]+400.*x[0].powi(3) - 2.*(1.-x[0]),
        -200.*(x[1]-x[0].powi(2))
    ]
}

fn hessienne_f2(x: &Array1<f64>) -> Array2<f64> {
    array![
        [-300.*x[1]+1200.*x[0].powi(2)+2., -3.*x[0]],
        [400.*x[0], -200.]
    ]
}

fn main() {
    let params = AlgoParams {
        epsilon: 1e-12,
        kMax: 500
    };

    // Newton

    println!("* f1");
    let x011 = array![1., 0., 0.];
    let x012 = array![10., 3., -2.2];

    println!("{} -> {:?}", x011, newton(gradient_f1, hessienne_f1, &x011, &params));
    println!("{} -> {:?}", x012, newton(gradient_f1, hessienne_f1, &x012, &params));

    println!("* f2");
    let x021 = array![-1.2, 1.];
    let x022 = array![10., 0.];
    let x023 = array![0., 1./200.+1e-12];

    println!("{} -> {:?}", x021, newton(gradient_f2, hessienne_f2, &x021, &params));
    println!("{} -> {:?}", x022, newton(gradient_f2, hessienne_f2, &x022, &params));
    println!("{} -> {:?}", x023, newton(gradient_f2, hessienne_f2, &x023, &params));

    // pas de cauchy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pas_de_cauchy() {
        let pas_cauchy_exemple1_g = array![0., 0.];
        let pas_cauchy_exemple1_H = array![[7., 0.], [0., 0.2]];
        let res1 = array![0., 0.];
        let pas_cauchy_exemple2_g = array![6., 2.];
        let pas_cauchy_exemple2_H = &pas_cauchy_exemple1_H;
        let res2 = -0.5/f64::sqrt(40.)*&pas_cauchy_exemple2_g;
        let pas_cauchy_exemple3_g = array![-2., 1.];
        let pas_cauchy_exemple3_H = array![[-2., 0.], [0., 10.]];
        let res3 = -0.5/f64::sqrt(5.)*&pas_cauchy_exemple3_g;


        assert!((super::pas_de_cauchy(&pas_cauchy_exemple1_g, &pas_cauchy_exemple1_H, 0.5, EPSILON) - &res1).norm() < EPSILON);
        assert!((super::pas_de_cauchy(&pas_cauchy_exemple2_g, &pas_cauchy_exemple2_H, 0.5, EPSILON) - &res2).norm() < EPSILON);
        assert!((super::pas_de_cauchy(&pas_cauchy_exemple3_g, &pas_cauchy_exemple3_H, 0.5, EPSILON) - &res3).norm() < EPSILON);

    }
}

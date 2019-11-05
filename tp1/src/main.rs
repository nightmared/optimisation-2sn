use ndarray::{Array, Array1, Array2, ArrayBase};
use ndarray_linalg::Norm;

use ndarray::prelude::*;
use ndarray_linalg::Solve;


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
}

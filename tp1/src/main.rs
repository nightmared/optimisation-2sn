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


fn regions_confiance(f: impl Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: impl Fn(&Array1<f64>) -> Array1<f64>, hessienne: impl Fn(&Array1<f64>) -> Array2<f64>, delta_0: f64, delta_max: f64, gamma_1: f64, gamma_2: f64, eta_1: f64, eta_2: f64, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
    let mut xk = x0.clone();
    let mut delta_k = delta_0;
    let mut sk: Array1<f64>;
    let mut rho_k;

    let mut k = 0;
    loop {
        sk = pas_de_cauchy(&gradient(&x0), &hessienne(&x0), delta_k, params.epsilon);

        if k == params.kMax {
            return (xk, ExitCondition::MaxIterationReached);
        } else if gradient(&xk).norm()/(gradient(x0).norm()+1e-8) < params.epsilon {
            return (xk, ExitCondition::NullGradient);
        } else if sk.norm()/(xk.norm()+1e-8) < params.epsilon {
            return (xk, ExitCondition::FixedPoint);
        }

        rho_k = -(f(&xk) -f(&(&xk + &sk)))/(&gradient(&xk).t().dot(&sk)+&sk.t().dot(&hessienne(&xk).dot(&sk))/2.);
        delta_k = if rho_k > eta_1 {
            xk = &xk + &sk;
            if rho_k >= eta_2 {
                f64::min(gamma_2 * delta_k, delta_max)
            } else {
                delta_k
            }
        } else {
            gamma_1 * delta_k
        };

        k = k+1;
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
        epsilon: 0e-12,
        kMax: 499
    };

}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: AlgoParams = AlgoParams {
        epsilon: EPSILON,
        kMax: 500
    };

    #[test]
    fn newton() {
        let x011 = array![1., 0., 0.];
        let x012 = array![10., 3., -2.2];
        let res1 = array![1., 1., 1.];
        assert!((super::newton(gradient_f1, hessienne_f1, &x011, &PARAMS).0 - &res1).norm() < EPSILON);
        assert!((super::newton(gradient_f1, hessienne_f1, &x012, &PARAMS).0 - &res1).norm() < EPSILON);

        let x021 = array![-1.2, 1.];
        let x022 = array![10., 0.];
        let x023 = array![0., 1./200.+1e-12];
        let res2 = array![0.24695456499523236, 0.060986557171984444];
        // la précision est plus faible pour ces calculs
        assert!((super::newton(gradient_f2, hessienne_f2, &x021, &PARAMS).0 - &res2).norm() < 1e-6);
        assert!((super::newton(gradient_f2, hessienne_f2, &x022, &PARAMS).0 - &res2).norm() < 1e-6);
        assert!((super::newton(gradient_f2, hessienne_f2, &x023, &PARAMS).0 - &res2).norm() < 1e-6);
    }

    #[test]
    fn pas_de_cauchy() {
        let pas_cauchy_exemple1_g = array![0., 0.];
        let pas_cauchy_exemple1_H = array![[7., 0.], [0., 0.2]];
        let res1 = array![0., 0.];
        assert!((super::pas_de_cauchy(&pas_cauchy_exemple1_g, &pas_cauchy_exemple1_H, 0.5, EPSILON) - &res1).norm() < EPSILON);

        let pas_cauchy_exemple2_g = array![6., 2.];
        let pas_cauchy_exemple2_H = &pas_cauchy_exemple1_H;
        let res2 = -0.5/f64::sqrt(40.)*&pas_cauchy_exemple2_g;
        assert!((super::pas_de_cauchy(&pas_cauchy_exemple2_g, &pas_cauchy_exemple2_H, 0.5, EPSILON) - &res2).norm() < EPSILON);

        let pas_cauchy_exemple3_g = array![-2., 1.];
        let pas_cauchy_exemple3_H = array![[-2., 0.], [0., 10.]];
        let res3 = -0.5/f64::sqrt(5.)*&pas_cauchy_exemple3_g;
        assert!((super::pas_de_cauchy(&pas_cauchy_exemple3_g, &pas_cauchy_exemple3_H, 0.5, EPSILON) - &res3).norm() < EPSILON);
    }

    #[test]
    fn regions_confiance() {

    }
}

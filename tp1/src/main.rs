use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;

use ndarray::prelude::*;
use ndarray_linalg::Solve;

#[derive(Debug)]
enum ExitCondition {
    MaxIterationReached,
    NullGradient,
    FixedPoint
}

#[derive(Debug, Default)]
struct AlgoParams {
    epsilon: f64,
    k_max: usize,
    delta_0: f64,
    delta_max: f64,
    gamma_1: f64,
    gamma_2: f64,
    eta_1: f64,
    eta_2: f64
}

// _f is only there to maintin a compatible interface with other methos and thus simplify tests
fn newton(_f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
    let mut xk = x0.clone();
    let mut dk = hessienne(&xk).solve_into(-gradient(&xk)).unwrap();

    let mut k = 0;
    loop {
        xk = xk+dk;
        k += 1;

        dk = hessienne(&xk).solve_into(-gradient(&xk)).unwrap();
        if k == params.k_max {
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


fn regions_confiance<C>(methode: &C, f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition)
    where C: Fn(&Array1<f64>, &Array2<f64>, f64, f64) -> Array1<f64> {
    let mut xk = x0.clone();
    let mut delta_k = params.delta_0;
    let mut sk: Array1<f64>;
    let mut rho_k;

    let mut k = 0;
    loop {
        sk = methode(&gradient(&xk), &hessienne(&xk), delta_k, params.epsilon);

        if k == params.k_max {
            return (xk, ExitCondition::MaxIterationReached);
        } else if gradient(&xk).norm()/(gradient(x0).norm()+1e-8) < params.epsilon {
            return (xk, ExitCondition::NullGradient);
        } else if sk.norm()/(xk.norm()+1e-8) < params.epsilon {
            return (xk, ExitCondition::FixedPoint);
        }

        rho_k = (f(&(&xk + &sk)) - f(&xk))/(&gradient(&xk).t().dot(&sk)+&sk.t().dot(&hessienne(&xk).dot(&sk))/2.);
        delta_k = if rho_k > params.eta_1 {
            xk = &xk + &sk;
            if rho_k >= params.eta_2 {
                f64::min(params.gamma_2 * delta_k, params.delta_max)
            } else {
                delta_k
            }
        } else {
            params.gamma_1 * delta_k
        };

        k = k+1;
    }
}

fn regions_confiance_pas_de_cauchy(f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
    regions_confiance(&pas_de_cauchy, f, x0, gradient, hessienne, params)
}

fn conjuge_tronque(gk: &Array1<f64>, hk: &Array2<f64>, delta_k: f64, epsilon: f64) -> Array1<f64> {
}

fn regions_confiance_conjuge_tronque(f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
    regions_confiance(&conjuge_tronque, f, x0, gradient, hessienne, params)
}

fn f1(x: &Array1<f64>) -> f64 {
    2.*(x[0]+x[1]+x[2]-3.).powi(2)+(x[0]-x[1]).powi(2)+(x[1]-x[2]).powi(2)
}

fn gradient_f1(x: &Array1<f64>) -> Array1<f64> {
    array![
        4.*(x[0]+x[1]+x[2]-3.) + 2.*(x[0]-x[1]),
        4.*(x[0]+x[1]+x[2]-3.) - 2.*(x[0]-x[1]) + 2.*(x[1]-x[2]),
        4.*(x[0]+x[1]+x[2]-3.) - 2.*(x[1]-x[2])
    ]
}

fn hessienne_f1(_x: &Array1<f64>) -> Array2<f64> {
    array![
        [6., 2., 4.],
        [2., 8., 2.],
        [4., 2., 6.]
    ]
}

fn f2(x: &Array1<f64>) -> f64 {
    100.*(x[1]-x[0].powi(2)).powi(2)+(1.-x[0]).powi(2)
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

}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: AlgoParams = AlgoParams {
        epsilon: 1e-14,
        k_max: 500,
        delta_0: 5e-4,
        delta_max: 5.0,
        gamma_1: 0.5,
        gamma_2: 1.5,
        eta_1: 0.1,
        eta_2: 0.8
    };

    fn test_annexe_a1<M>(method: M, f: &dyn Fn(&Array1<f64>) -> f64, g: &dyn Fn(&Array1<f64>) -> Array1<f64>, h: &dyn Fn(&Array1<f64>) -> Array2<f64>)
    where M: Fn(&dyn Fn(&Array1<f64>) -> f64, &Array1<f64>, &dyn Fn(&Array1<f64>) -> Array1<f64>, &dyn Fn(&Array1<f64>) -> Array2<f64>, &AlgoParams) -> (Array1<f64>, ExitCondition) {
        let x011 = array![1., 0., 0.];
        let x012 = array![10., 3., -2.2];
        let res = array![1., 1., 1.];
        assert!((method(&f, &x011, &g, &h, &PARAMS).0 - &res).norm() < 1e-10);
        assert!((method(&f, &x012, &g, &h, &PARAMS).0 - &res).norm() < 1e-10);
    }


    fn test_annexe_a2<M>(method: M, f: &dyn Fn(&Array1<f64>) -> f64, g: &dyn Fn(&Array1<f64>) -> Array1<f64>, h: &dyn Fn(&Array1<f64>) -> Array2<f64>)
     where M: Fn(&dyn Fn(&Array1<f64>) -> f64, &Array1<f64>, &dyn Fn(&Array1<f64>) -> Array1<f64>, &dyn Fn(&Array1<f64>) -> Array2<f64>, &AlgoParams) -> (Array1<f64>, ExitCondition) {
        let x021 = array![-1.2, 1.];
        let x022 = array![10., 0.];
        let x023 = array![0., 1./200.+1e-12];
        let res = array![0.24695456499523236, 0.060986557171984444];
        // la précision est plus faible pour ces calculs
        assert!((method(&f, &x021, &g, &h, &PARAMS).0 - &res).norm() < 1e-7);
        assert!((method(&f, &x022, &g, &h, &PARAMS).0 - &res).norm() < 1e-7);
        assert!((method(&f, &x023, &g, &h, &PARAMS).0 - &res).norm() < 1e-7);

    }

    #[test]
    fn newton() {
        test_annexe_a1(&super::newton, &f1, &gradient_f1, &hessienne_f1);
        test_annexe_a2(&super::newton, &f2, &gradient_f2, &hessienne_f2);
    }

    #[test]
    fn pas_de_cauchy() {
        let pas_cauchy_exemple1_g = array![0., 0.];
        let pas_cauchy_exemple1_h = array![[7., 0.], [0., 0.2]];
        let res1 = array![0., 0.];
        assert!((super::pas_de_cauchy(&pas_cauchy_exemple1_g, &pas_cauchy_exemple1_h, 0.5, 1e-14) - &res1).norm() < PARAMS.epsilon);

        let pas_cauchy_exemple2_g = array![6., 2.];
        let pas_cauchy_exemple2_h = &pas_cauchy_exemple1_h;
        let res2 = -0.5/f64::sqrt(40.)*&pas_cauchy_exemple2_g;
        assert!((super::pas_de_cauchy(&pas_cauchy_exemple2_g, &pas_cauchy_exemple2_h, 0.5, 1e-14) - &res2).norm() < PARAMS.epsilon);

        let pas_cauchy_exemple3_g = array![-2., 1.];
        let pas_cauchy_exemple3_h = array![[-2., 0.], [0., 10.]];
        let res3 = -0.5/f64::sqrt(5.)*&pas_cauchy_exemple3_g;
        assert!((super::pas_de_cauchy(&pas_cauchy_exemple3_g, &pas_cauchy_exemple3_h, 0.5, 1e-14) - &res3).norm() < PARAMS.epsilon);
    }

    #[test]
    fn regions_confiance_pas_de_cauchy() {
        test_annexe_a1(&super::regions_confiance_pas_de_cauchy, &f1, &gradient_f1, &hessienne_f1);
        test_annexe_a2(&super::regions_confiance_pas_de_cauchy, &f2, &gradient_f2, &hessienne_f2);
    }

    #[test]
    fn regions_confiance_conjuge_tronque() {
        test_annexe_a1(&super::regions_confiance_conjuge_tronque, &f1, &gradient_f1, &hessienne_f1);
        test_annexe_a2(&super::regions_confiance_conjuge_tronque, &f2, &gradient_f2, &hessienne_f2);
    }
}

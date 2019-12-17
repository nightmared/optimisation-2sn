use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;

use ndarray::prelude::*;
use ndarray_linalg::Solve;

#[derive(Debug)]
enum ExitState {
    MaxIterationReached,
    NullGradient,
    FixedPoint,
    FunStable
}

#[derive(Debug)]
struct ExitCondition(ExitState, usize);

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

// _f is only there to maintain a compatible interface with other methods and thus simplify tests
fn newton(f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
    let mut xk = x0.clone();
    let mut dk = -hessienne(&xk).solve_into(gradient(&xk)).unwrap();

    let mut k = 0;
    let mut fk = f(&xk);
    loop {
        xk = xk+dk;
        k += 1;

        dk = -hessienne(&xk).solve_into(gradient(&xk)).unwrap();
        if k == params.k_max {
            return (xk, ExitCondition(ExitState::MaxIterationReached, k));
        } else if gradient(&xk).norm() < params.epsilon * (gradient(x0).norm()+1e-8) {
            return (xk, ExitCondition(ExitState::NullGradient, k));
        } else if dk.norm() < params.epsilon * (xk.norm()+1e-8) {
            return (xk, ExitCondition(ExitState::FixedPoint, k));
        } else if (f(&xk)-fk).abs() < params.epsilon * (fk.abs()+1e-8) {
            return (xk, ExitCondition(ExitState::FunStable, k));
        }

        fk = f(&xk);
    }
}

fn pas_de_cauchy(gk: Array1<f64>, hk: &Array2<f64>, delta_k: f64, epsilon: f64) -> Array1<f64> {
    if gk.norm() == 0. {
        return Array1::zeros(gk.len());
    }

    let a = gk.t().dot(&hk.dot(&gk))/2.;
    let b = -gk.norm().powi(2);
    let t_max = delta_k/gk.norm();

    if a <= 0. {
        -t_max*gk
    } else {
        -f64::min(-b/(2.*a), t_max)*gk
    }
}


fn regions_confiance<C>(methode: &C, f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition)
    where C: Fn(Array1<f64>, &Array2<f64>, f64, f64) -> Array1<f64> {
    let mut xk = x0.clone();
    let mut delta_k = params.delta_0;
    let mut sk: Array1<f64>;

    let mut k = 0;
    let mut fk = f(&xk);
    loop {
        sk = methode(gradient(&xk), &hessienne(&xk), delta_k, params.epsilon);

        let xk_plus_1 = &xk + &sk;
        let fk_plus_1 = f(&xk_plus_1);
        let rho_k = (fk_plus_1 - fk)/(&gradient(&xk).t().dot(&sk)+&sk.t().dot(&hessienne(&xk).dot(&sk))/2.);
        delta_k = if rho_k > params.eta_1 {
            xk = xk_plus_1;
            if (fk_plus_1-fk).abs() < params.epsilon * (fk.abs()+1e-8) {
                return (xk, ExitCondition(ExitState::FunStable, k));
            }
            fk = fk_plus_1;
            if rho_k >= params.eta_2 {
                f64::min(params.gamma_2 * delta_k, params.delta_max)
            } else {
                delta_k
            }
        } else {
            params.gamma_1 * delta_k
        };

        k = k+1;

        if rho_k > params.eta_1 {
            if k == params.k_max {
                return (xk, ExitCondition(ExitState::MaxIterationReached, k));
            } else if gradient(&xk).norm() < params.epsilon * (gradient(x0).norm()+1e-8) {
                return (xk, ExitCondition(ExitState::NullGradient, k));
            } else if sk.norm() < params.epsilon * (xk.norm()+1e-8) {
                return (xk, ExitCondition(ExitState::FixedPoint, k));
            }
        }

    }
}

fn regions_confiance_pas_de_cauchy(f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
    regions_confiance(&pas_de_cauchy, f, x0, gradient, hessienne, params)
}


fn get_min_root(g: &Array1<f64>, h: &Array2<f64>, s: &Array1<f64>, p: &Array1<f64>, delta_k: f64) -> Array1<f64> {
    let s_square = s.t().dot(s);
    let p_square = p.t().dot(p);
    let delta = 4.*(1.-&s.t().dot(p)*(&s_square-delta_k.powi(2)));
    let rho_1 = (-2.*s_square-delta.sqrt())/(2.*p_square);
    let rho_2 = (-2.*s_square+delta.sqrt())/(2.*p_square);

    // détermination du minimum
    let rho_p_1 = rho_1*p;
    let rho_p_2 = rho_2*p;
    if g.t().dot(&rho_p_1) + rho_p_1.t().dot(&h.dot(&rho_p_1))/2.
        < g.t().dot(&rho_p_2) + rho_p_2.t().dot(&h.dot(&rho_p_2))/2. {
            rho_p_1
    } else {
            rho_p_2
    }
}

fn get_pos_root(s: &Array1<f64>, p: &Array1<f64>, delta_k: f64) -> f64 {
    let s_square = s.t().dot(s);
    let p_square = p.t().dot(p);
    let delta = 4.*(1.-&s.t().dot(p)*(&s_square-delta_k.powi(2)));
    let rho_1 = (-2.*s_square-delta.sqrt())/(2.*p_square);
    let rho_2 = (-2.*s_square+delta.sqrt())/(2.*p_square);

    // détermination du minimum
    if rho_1 > 0. {
        rho_1
    } else {
        rho_2
    }
}


fn conjuge_tronque(gk: Array1<f64>, hk: &Array2<f64>, delta_k: f64, epsilon: f64) -> Array1<f64> {
    let mut s = Array1::<f64>::zeros(gk.len());
    let mut p = -&gk;
    let mut g = gk;

    let mut k = 0;
    loop {
        let kappa = p.t().dot(&hk.dot(&p));
        if kappa < 0. {
            return &s + &get_min_root(&g, hk, &s, &p, delta_k);
        }

        let alpha = &g.t().dot(&g)/kappa;

        if (&s+&(alpha*&p)).norm() >= delta_k {
            return &s + &(get_pos_root(&s, &p, delta_k)*&p);
        }

        s = s + alpha * &p;

        let new_g = &g + &(alpha * &hk.dot(&p));

        let beta = &new_g.t().dot(&new_g)/&g.t().dot(&g);

        g = new_g;

        p = -&g + beta * p;


        k += 1;
    }
    return p;
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
        -400.*x[0]*(x[1]-x[0].powi(2)) - 2.*(1.-x[0]),
        200.*(x[1]-x[0].powi(2))
    ]
}

fn hessienne_f2(x: &Array1<f64>) -> Array2<f64> {
    array![
        [1200.*x[0].powi(2)-400.*x[1]+2., -400.*x[0]],//400.*(x[0].powi(2)-2.*x[1])],
        [-400.*x[0], 200.]
    ]
}

fn main() {

}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: AlgoParams = AlgoParams {
        epsilon: 1e-8,
        k_max: 5000,
        delta_0: 1.0,
        delta_max: 1e8,
        gamma_1: 0.5,
        gamma_2: 2.0,
        eta_1: 0.1,
        eta_2: 0.5
    };

    fn test_annexe_a1<M>(method: M, f: &dyn Fn(&Array1<f64>) -> f64, g: &dyn Fn(&Array1<f64>) -> Array1<f64>, h: &dyn Fn(&Array1<f64>) -> Array2<f64>, eps: f64, params: &AlgoParams)
    where M: Fn(&dyn Fn(&Array1<f64>) -> f64, &Array1<f64>, &dyn Fn(&Array1<f64>) -> Array1<f64>, &dyn Fn(&Array1<f64>) -> Array2<f64>, &AlgoParams) -> (Array1<f64>, ExitCondition) {
        let x011 = array![1., 0., 0.];
        let x012 = array![10., 3., -2.2];
        let res = array![1., 1., 1.];
        assert!((method(&f, &x011, &g, &h, params).0 - &res).norm() < eps);
        assert!((method(&f, &x012, &g, &h, params).0 - &res).norm() < eps);
    }


    fn test_annexe_a2<M>(method: M, f: &dyn Fn(&Array1<f64>) -> f64, g: &dyn Fn(&Array1<f64>) -> Array1<f64>, h: &dyn Fn(&Array1<f64>) -> Array2<f64>, eps: f64, params: &AlgoParams)
     where M: Fn(&dyn Fn(&Array1<f64>) -> f64, &Array1<f64>, &dyn Fn(&Array1<f64>) -> Array1<f64>, &dyn Fn(&Array1<f64>) -> Array2<f64>, &AlgoParams) -> (Array1<f64>, ExitCondition) {
        let x021 = array![-1.2, 1.];
        let x022 = array![10., 0.];
        let x023 = array![0., 1./200.+1e-12];
        let res = array![1.0, 1.0];
        assert!((method(&f, &x021, &g, &h, params).0 - &res).norm() < eps);
        assert!((method(&f, &x022, &g, &h, params).0 - &res).norm() < eps);
        //assert!((method(&f, &x023, &g, &h, &params).0 - &res).norm() < eps);

    }

    #[test]
    fn newton() {
        test_annexe_a1(&super::newton, &f1, &gradient_f1, &hessienne_f1, 1e-6, &PARAMS);
        // la précision est plus faible pour ces calculs
        test_annexe_a2(&super::newton, &f2, &gradient_f2, &hessienne_f2, 1e-2, &PARAMS);
    }

    #[test]
    fn pas_de_cauchy() {
        let pas_cauchy_exemple1_g = array![0., 0.];
        let pas_cauchy_exemple1_h = array![[7., 0.], [0., 0.2]];
        let res1 = array![0., 0.];
        assert!((super::pas_de_cauchy(pas_cauchy_exemple1_g, &pas_cauchy_exemple1_h, 0.5, 1e-14) - &res1).norm() < PARAMS.epsilon);

        let pas_cauchy_exemple2_g = array![6., 2.];
        let pas_cauchy_exemple2_h = &pas_cauchy_exemple1_h;
        let res2 = -0.5/f64::sqrt(40.)*&pas_cauchy_exemple2_g;
        assert!((super::pas_de_cauchy(pas_cauchy_exemple2_g, &pas_cauchy_exemple2_h, 0.5, 1e-14) - &res2).norm() < PARAMS.epsilon);

        let pas_cauchy_exemple3_g = array![-2., 1.];
        let pas_cauchy_exemple3_h = array![[-2., 0.], [0., 10.]];
        let res3 = -0.5/f64::sqrt(5.)*&pas_cauchy_exemple3_g;
        //println!("{} {} {}",  res1, res2, res3);
        assert!((super::pas_de_cauchy(pas_cauchy_exemple3_g, &pas_cauchy_exemple3_h, 0.5, 1e-14) - &res3).norm() < PARAMS.epsilon);
    }

    #[test]
    fn regions_confiance_pas_de_cauchy() {
        let mut params = PARAMS;
        params.epsilon = 1e-4;
        test_annexe_a1(&super::regions_confiance_pas_de_cauchy, &f1, &gradient_f1, &hessienne_f1, 1e-2, &params);
        test_annexe_a2(&super::regions_confiance_pas_de_cauchy, &f2, &gradient_f2, &hessienne_f2, 0.2, &params);
    }

    #[test]
    fn regions_confiance_conjuge_tronque() {
        //test_annexe_a1(&super::regions_confiance_conjuge_tronque, &f1, &gradient_f1, &hessienne_f1);
        //test_annexe_a2(&super::regions_confiance_conjuge_tronque, &f2, &gradient_f2, &hessienne_f2);
    }
}

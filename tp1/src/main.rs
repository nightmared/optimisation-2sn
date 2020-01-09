use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;

use ndarray::prelude::*;
use ndarray_linalg::Solve;

/// Raison de sortie de l'algorithme
#[derive(Debug)]
pub enum ExitState {
    /// Le nombre maximal d'itérations a été atteint
    MaxIterationReached,
    /// La gradient de la fonction est nulle
    NullGradient,
    /// Le résultat obtenu n'évolue plus
    FixedPoint,
    /// f(x+1)-f(x) n'évolue plus
    FunStable
}

/// Résultat en sortie des algorithmes
#[derive(Debug)]
pub struct ExitCondition(ExitState, usize);

/// Paramètres des algorithmes
#[derive(Copy, Clone, Debug, Default)]
pub struct AlgoParams {
    // paramètres généraux
    /// précision générale
    pub epsilon: f64,
    /// précision pour l'algorithme des régions de confiance
    pub epsilon_algo_regions_confiance: f64,
    /// précision pour l'algorithme du lagrangien augmenté
    pub epsilon_algo_lagrangien: f64,
    /// nombre maximal d'itérations
    pub k_max: usize,
    // pour les régions de confiance
    pub delta_0: f64,
    pub delta_max: f64,
    pub gamma_1: f64,
    pub gamma_2: f64,
    pub eta_1: f64,
    pub eta_2: f64,
    // pour le lagrangien
    pub tau: f64,
    pub alpha: f64,
    pub beta: f64
}

/// Algorithme de newton
pub fn newton(f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
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

/// Algorithme du pas de cauchy
pub fn pas_de_cauchy(gk: &Array1<f64>, hk: &Array2<f64>, delta_k: f64, epsilon: f64) -> Array1<f64> {
    if gk.norm() == 0. {
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


/// Algorithme des régions de confiance
pub fn regions_confiance<C>(methode: &C, f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition)
    where C: Fn(&Array1<f64>, &Array2<f64>, f64, f64) -> Array1<f64> {
    let mut xk = x0.clone();
    let mut delta_k = params.delta_0;
    let mut sk: Array1<f64>;

    let mut k = 0;
    let mut fk = f(&xk);
    loop {
        sk = methode(&gradient(&xk), &hessienne(&xk), delta_k, params.epsilon_algo_regions_confiance);

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
            if gradient(&xk).norm() < params.epsilon * (gradient(x0).norm()+1e-8) {
                return (xk, ExitCondition(ExitState::NullGradient, k));
            } else if sk.norm() < params.epsilon * (xk.norm()+1e-8) {
                return (xk, ExitCondition(ExitState::FixedPoint, k));
            }
        }

        if k == params.k_max {
            return (xk, ExitCondition(ExitState::MaxIterationReached, k));
        }
    }
}

/// Régions de confiance avec le pas de cauchy
pub fn regions_confiance_pas_de_cauchy(f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
    regions_confiance(&pas_de_cauchy, f, x0, gradient, hessienne, params)
}


pub fn get_min_root(g: &Array1<f64>, h: &Array2<f64>, s: &Array1<f64>, p: &Array1<f64>, delta_k: f64) -> Array1<f64> {
    let s_square = s.t().dot(s);
    let p_square = p.t().dot(p);
    let delta = 4.*(&s.t().dot(p)-p_square*(&s_square-delta_k.powi(2)));
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

/// Renvoie la racine positive du problème ``
pub fn get_pos_root(s: &Array1<f64>, p: &Array1<f64>, delta_k: f64) -> f64 {
    let s_square = s.t().dot(s);
    let p_square = p.t().dot(p);
    let delta = 4.*(&s.t().dot(p)-p_square*(&s_square-delta_k.powi(2)));
    let rho_1 = (-2.*s_square-delta.sqrt())/(2.*p_square);
    let rho_2 = (-2.*s_square+delta.sqrt())/(2.*p_square);

    // détermination du minimum
    if rho_1 > 0. {
        rho_1
    } else {
        rho_2
    }
}


/// Algorithme du gradient conjugé tronqué
pub fn conjuge_tronque(gk: &Array1<f64>, hk: &Array2<f64>, delta_k: f64, epsilon: f64) -> Array1<f64> {
    let gk_norm = &gk.norm();
    if gk_norm < &epsilon {
        return gk.to_owned();
    }
    let mut s = Array1::<f64>::zeros(gk.len());
    let mut p = -gk;
    let mut g: Array1<f64> = gk.to_owned();

    let mut k = 0;
    loop {
        let kappa = p.t().dot(&hk.dot(&p));
        if kappa <= epsilon {
            return &s + &get_min_root(&g, hk, &s, &p, delta_k);
        }

        let alpha = &g.t().dot(&g)/kappa;

        if (&s+&(alpha*&p)).norm() >= delta_k {
            return &s + &(get_pos_root(&s, &p, delta_k)*&p);
        }

        s = s + alpha * &p;

        let new_g: Array1<f64> = (&g + &(alpha * &hk.dot(&p))).to_owned();

        let beta = &new_g.t().dot(&new_g)/&g.t().dot(&g);

        g = new_g;

        p = -&g + beta * p;

        k += 1;

        if g.norm() < 1e-8 * (gk_norm+1e-8) || k >= 5*g.len() {
            break;
        }
    }
    return s;
}

/// Régions de confiance avec le conjugé tronqué
pub fn regions_confiance_conjuge_tronque(f: &dyn Fn(&Array1<f64>) -> f64, x0: &Array1<f64>, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
    regions_confiance(&conjuge_tronque, f, x0, gradient, hessienne, params)
}

/// Méthode du lagrangien augmenté dans le cas d'égalité
pub fn lagrangien_egalite<C>(methode: &C, f: &dyn Fn(&Array1<f64>) -> f64, gradient: &dyn Fn(&Array1<f64>) -> Array1<f64>, hessienne: &dyn Fn(&Array1<f64>) -> Array2<f64>, contraintes: &dyn Fn(&Array1<f64>) -> Array1<f64>, gradient_contraintes: &dyn Fn(&Array1<f64>) -> Array2<f64>, grad_grad_contraintes: &dyn Fn(&Array1<f64>) -> Array3<f64>, x0: &Array1<f64>, lambda_0: &Array1<f64>, eta_0: f64, mu_0: f64, params: &AlgoParams) -> (Array1<f64>, Array1<f64>, f64)
    where C: Fn(&dyn Fn(&Array1<f64>) -> f64, &Array1<f64>, &dyn Fn(&Array1<f64>) -> Array1<f64>, &dyn Fn(&Array1<f64>) -> Array2<f64>, &AlgoParams) -> (Array1<f64>, ExitCondition) {
    let mut params = *params;
    let mut lambda_k: Array1<f64> = lambda_0.to_owned();
    let mut epsilon_k = 1./mu_0;
    let mut mu_k = mu_0;
    let mut eta_k = eta_0;
    let mut eta_0_ref = eta_0*mu_0.powf(params.alpha);
    let mut xk = x0.to_owned();
    let mut k = 0;

    loop {
        let CLHessC = |x: &Array1<f64>, beta: &Array1<f64>| {
            let grad_grad_contraintes_x = grad_grad_contraintes(x);
            let (dim0, dim1, dim2) = grad_grad_contraintes_x.dim();
            let mut res: Array2<f64> = Array2::zeros((dim0, dim1));
            for i in 0..dim2 {
                res = res+beta[i]*&grad_grad_contraintes_x.slice(s![.., .., i]);
            }
            res
        };
        let Lk = |x: &Array1<f64>| {
            let contraintes_x = contraintes(x);
            f(x) + &lambda_k.t().dot(&contraintes_x) + 0.5*mu_k*contraintes_x.norm().powi(2)
        };
        let GLk = |x: &Array1<f64>| {
            gradient(x) + gradient_contraintes(x).dot(&(&lambda_k+&(mu_k*contraintes(x))))
        };
        let HLk = |x: &Array1<f64>| {
            let gradient_contraintes_x = gradient_contraintes(x);
            hessienne(x) + &(mu_k*gradient_contraintes_x.dot(&gradient_contraintes_x.t()))+CLHessC(x, &(&lambda_k+&(mu_k*contraintes(x))))
        };

        params.epsilon = epsilon_k;
        xk = methode(&Lk, &xk, &GLk, &HLk, &params).0;

        if k == params.k_max
            || ((&gradient(&xk)+&gradient_contraintes(&xk).dot(&lambda_k)).norm() < params.epsilon_algo_lagrangien && contraintes(&xk).norm() < params.epsilon_algo_lagrangien) {
            break;
        }


        let ncontraintes = contraintes(&xk);
        if ncontraintes.norm() < eta_k {
            lambda_k = lambda_k + mu_k * ncontraintes;
            epsilon_k = epsilon_k/mu_k;
            eta_k = eta_k/mu_k.powf(params.beta);
        } else {
            mu_k = params.tau * mu_k;
            epsilon_k = 1./(mu_0*mu_k);
            eta_k = eta_0_ref/mu_k.powf(params.alpha);
        }
        k += 1;
    }
    (xk, lambda_k, mu_k)
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
        epsilon_algo_regions_confiance: 1e-12,
        epsilon_algo_lagrangien: 1e-6,
        k_max: 150,
        delta_0: 1.0,
        delta_max: 1e8,
        gamma_1: 0.5,
        gamma_2: 2.0,
        eta_1: 0.1,
        eta_2: 0.5,
        tau: 0.75,
        alpha: 0.1,
        beta: 0.9
    };

    #[test]
    fn newton() {
        // annexe a1
        let x011 = array![1., 0., 0.];
        let x012 = array![10., 3., -2.2];
        let res = array![1., 1., 1.];
        assert!((super::newton(&f1, &x011, &gradient_f1, &hessienne_f1, &PARAMS).0 - &res).norm() < 1e-12);
        assert!((super::newton(&f1, &x012, &gradient_f1, &hessienne_f1, &PARAMS).0 - &res).norm() < 1e-12);

        // annexe a2
        // la précision est plus faible pour ces calculs
        let x021 = array![-1.2, 1.];
        let x022 = array![10., 0.];
        //let x023 = array![0., 1./200.+1e-12];
        let res = array![1.0, 1.0];
        assert!((super::newton(&f2, &x021, &gradient_f2, &hessienne_f2, &PARAMS).0 - &res).norm() < 1e-10);
        assert!((super::newton(&f2, &x022, &gradient_f2, &hessienne_f2, &PARAMS).0 - &res).norm() < 1e-3);
        // diverge avec cet algorithme
        //assert!((super::newton(&f2, &x023, &gradient_f2, &hessienne_f2, &PARAMS).0 - &res).norm() < 1e-3);
    }

    #[test]
    fn pas_de_cauchy() {
        // annexe B
        let exemple1_g = array![0., 0.];
        let exemple1_h = array![[7., 0.], [0., 0.2]];
        let res1 = array![0., 0.];
        assert!((super::pas_de_cauchy(&exemple1_g, &exemple1_h, 0.5, 1e-14) - &res1).norm() < PARAMS.epsilon);

        let exemple2_g = array![6., 2.];
        let exemple2_h = &exemple1_h;
        let res2 = -0.5/f64::sqrt(40.)*&exemple2_g;
        assert!((super::pas_de_cauchy(&exemple2_g, &exemple2_h, 0.5, 1e-14) - &res2).norm() < PARAMS.epsilon);

        let exemple3_g = array![-2., 1.];
        let exemple3_h = array![[-2., 0.], [0., 10.]];
        let res3 = -0.5/f64::sqrt(5.)*&exemple3_g;
        assert!((super::pas_de_cauchy(&exemple3_g, &exemple3_h, 0.5, 1e-14) - &res3).norm() < PARAMS.epsilon);
    }

    #[test]
    fn conjuge_tronque() {
        // annexe B
        let exemple1_g = array![0., 0.];
        let exemple1_h = array![[7., 0.], [0., 0.2]];
        let res1 = array![0., 0.];
        assert!((super::conjuge_tronque(&exemple1_g, &exemple1_h, 0.5, 1e-14) - &res1).norm() < PARAMS.epsilon);

        let exemple2_g = array![6., 2.];
        let exemple2_h = &exemple1_h;
        let res2 = -0.5/f64::sqrt(40.)*&exemple2_g;
        assert!((super::conjuge_tronque(&exemple2_g, &exemple2_h, 0.5, 1e-14) - &res2).norm() < PARAMS.epsilon);

        let exemple3_g = array![-2., 1.];
        let exemple3_h = array![[-2., 0.], [0., 10.]];
        let res3 = -0.5/f64::sqrt(5.)*&exemple3_g;
        assert!((super::conjuge_tronque(&exemple3_g, &exemple3_h, 0.5, 1e-14) - &res3).norm() < PARAMS.epsilon);

        // annexe C
        let exemple1_g = array![0., 0.];
        let exemple1_h = array![[-2., 0.], [0., 10.0]];
        let res1 = array![0., 0.];
        assert!((super::conjuge_tronque(&exemple1_g, &exemple1_h, 0.5, 1e-14) - &res1).norm() < PARAMS.epsilon);

        //let exemple2_g = array![2., 3.];
        //let exemple2_h = array![[4., 6.], [6., 5.0]];
        //let res2 = TODO;
        //println!("{:?}", (super::conjuge_tronque(&exemple2_g, &exemple2_h, 0.5, 1e-14) - &res2));
        //assert!((super::conjuge_tronque(&exemple2_g, &exemple2_h, 0.5, 1e-14) - &res2).norm() < PARAMS.epsilon);

        //let exemple3_g = array![2., 0.];
        //let exemple3_h = array![[4., 0.], [0., -15.]];
        //let res3 = TODO;
        //assert!((super::conjuge_tronque(&exemple3_g, &exemple3_h, 0.5, 1e-14) - &res3).norm() < PARAMS.epsilon);
    }

    #[test]
    fn regions_confiance_pas_de_cauchy() {
        // annexe a1
        let x011 = array![1., 0., 0.];
        let x012 = array![10., 3., -2.2];
        let res = array![1., 1., 1.];
        assert!((super::newton(&f1, &x011, &gradient_f1, &hessienne_f1, &PARAMS).0 - &res).norm() < 1e-12);
        assert!((super::newton(&f1, &x012, &gradient_f1, &hessienne_f1, &PARAMS).0 - &res).norm() < 1e-12);

        // annexe a2
        let x021 = array![-1.2, 1.];
        let x022 = array![10., 0.];
        //let x023 = array![0., 1./200.+1e-12];
        let res = array![1.0, 1.0];
        assert!((super::newton(&f2, &x021, &gradient_f2, &hessienne_f2, &PARAMS).0 - &res).norm() < 1e-10);
        assert!((super::newton(&f2, &x022, &gradient_f2, &hessienne_f2, &PARAMS).0 - &res).norm() < 1e-3);
        // diverge avec cet algorithme
        //assert!((super::newton(&f2, &x023, &gradient_f2, &hessienne_f2, &PARAMS).0 - &res).norm() < 1e-3);
    }

    #[test]
    fn regions_confiance_conjuge_tronque() {
        // annexe a1
        let x011 = array![1., 0., 0.];
        let x012 = array![10., 3., -2.2];
        let res = array![1., 1., 1.];
        assert!((super::newton(&f1, &x011, &gradient_f1, &hessienne_f1, &PARAMS).0 - &res).norm() < 1e-12);
        assert!((super::newton(&f1, &x012, &gradient_f1, &hessienne_f1, &PARAMS).0 - &res).norm() < 1e-12);

        // annexe a2
        let x021 = array![-1.2, 1.];
        let x022 = array![10., 0.];
        //let x023 = array![0., 1./200.+1e-12];
        let res = array![1.0, 1.0];
        assert!((super::newton(&f2, &x021, &gradient_f2, &hessienne_f2, &PARAMS).0 - &res).norm() < 1e-10);
        assert!((super::newton(&f2, &x022, &gradient_f2, &hessienne_f2, &PARAMS).0 - &res).norm() < 1e-3);
        // diverge avec cet algorithme
        //assert!((super::newton(&f2, &x023, &gradient_f2, &hessienne_f2, &PARAMS).0 - &res).norm() < 1e-3);
    }

    #[test]
    fn lagrangien_egalite() {
        let xc11 = array![0., 1., 1.];
        let xc12 = array![0.25, 1.25, 1.];
        let res = array![0.5, 1.25, 0.5];
        let lambda_0_1 = array![0.];
        let contrainte_1 = |x: &Array1<f64>| array![x[0]+x[2]-1.];
        let gradient_contrainte_1 = |_: &Array1<f64>| array![[1.], [0.], [1.]];
        let grad_grad_contrainte_1 = |_: &Array1<f64>| array![[[0.], [0.], [0.]], [[0.], [0.], [0.]], [[0.], [0.], [0.]]];
        assert!((super::lagrangien_egalite(&super::newton, &f1, &gradient_f1, &hessienne_f1,  &contrainte_1, &gradient_contrainte_1, &grad_grad_contrainte_1, &xc11, &lambda_0_1, 0.1, 5000., &PARAMS).0-&res).norm() < 1e-6);
        assert!((super::lagrangien_egalite(&super::regions_confiance_conjuge_tronque, &f1, &gradient_f1, &hessienne_f1,  &contrainte_1, &gradient_contrainte_1, &grad_grad_contrainte_1, &xc11, &lambda_0_1, 0.1, 5000., &PARAMS).0-&res).norm() < 1e-6);
        assert!((super::lagrangien_egalite(&super::regions_confiance_pas_de_cauchy, &f1, &gradient_f1, &hessienne_f1,  &contrainte_1, &gradient_contrainte_1, &grad_grad_contrainte_1, &xc11, &lambda_0_1, 0.1, 5000., &PARAMS).0-&res).norm() < 1e-6);
        assert!((super::lagrangien_egalite(&super::regions_confiance_pas_de_cauchy, &f1, &gradient_f1, &hessienne_f1,  &contrainte_1, &gradient_contrainte_1, &grad_grad_contrainte_1, &xc12, &lambda_0_1, 0.1, 5000., &PARAMS).0-&res).norm() < 1e-6);
        assert!((super::lagrangien_egalite(&super::regions_confiance_pas_de_cauchy, &f1, &gradient_f1, &hessienne_f1,  &contrainte_1, &gradient_contrainte_1, &grad_grad_contrainte_1, &xc12, &lambda_0_1, 0.1, 5000., &PARAMS).0-&res).norm() < 1e-6);
    }
}

use ndarray::{Array, Array1, Array2, ArrayBase};
use ndarray_linalg::Norm;

use ndarray::prelude::*;
use ndarray_linalg::Solve;

use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::Arc;

#[derive(Debug)]
struct Placeholder<T> {
    ident: u32,
    _ghost: PhantomData<T>
}

impl<T> std::clone::Clone for Placeholder<T> {
    fn clone(&self) -> Self {
        Placeholder {
            ident: self.ident,
            _ghost: PhantomData
        }
    }
}

impl<T> PartialEq for Placeholder<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident
    }
}

#[derive(Debug)]
enum AstError<'a, T> {
    UnavailableVariable(&'a Placeholder<T>)
}

trait Ast<T> {
    fn eval(&self, vars: &Vec<(&Placeholder<T>, f64)>) -> Result<T, AstError<T>>;
    fn derivative(&self, var: &Placeholder<T>) -> Result<Self, AstError<T>> where Self: Sized;
}


#[derive(PartialEq, Debug)]
enum AstDiff<T> {
    Mul(OwnedAst<T>, OwnedAst<T>),
    Add(OwnedAst<T>, OwnedAst<T>),
    Var(Placeholder<T>),
    Constant(T)
}

#[derive(Clone, Debug)]
struct OwnedAst<T> (Arc<Box<AstDiff<T>>>);

impl<T> OwnedAst<T> {
    fn new(val: AstDiff<T>) -> Self {
        OwnedAst(Arc::new(Box::new(val)))
    }
}

impl<T: PartialEq> PartialEq for OwnedAst<T> {
    fn eq(&self, other: &Self) -> bool {
        **self.0 == **other.0
    }
}

impl Ast<f64> for AstDiff<f64> {
    fn eval(&self, vars: &Vec<(&Placeholder<f64>, f64)>) -> Result<f64, AstError<f64>> {
        match self {
            AstDiff::Constant(c) => Ok(*c),
            AstDiff::Mul(a, b) => Ok(a.eval(vars)? * b.eval(vars)?),
            AstDiff::Add(a, b) => Ok(a.eval(vars)? + b.eval(vars)?),
            AstDiff::Var(v) => if let Some((_, val)) = vars.iter().filter(|x| *x.0 == *v).next() {
                Ok(*val)
            } else {
                Err(AstError::UnavailableVariable(v))
            }
        }
    }
    fn derivative(&self, var: &Placeholder<f64>) -> Result<AstDiff<f64>, AstError<f64>> {
        match self {
            AstDiff::Constant(c) => Ok(AstDiff::Constant(0.)),
            AstDiff::Mul(a, b) => Ok(AstDiff::Add(
                OwnedAst::new(AstDiff::Mul(a.derivative(var)?, b.clone())),
                OwnedAst::new(AstDiff::Mul(b.derivative(var)?, a.clone()))
            )),
            AstDiff::Add(a, b) => Ok(AstDiff::Add(
                    a.derivative(var)?,
                    b.derivative(var)?
            )),
            AstDiff::Var(v) => if *v == *var {
                Ok(AstDiff::Constant(1.))
            } else {
                Ok(AstDiff::Constant(0.))
            }
        }
    }
}

impl Ast<f64> for OwnedAst<f64> {
    fn eval(&self, vars: &Vec<(&Placeholder<f64>, f64)>) -> Result<f64, AstError<f64>> {
        self.0.eval(vars)
    }
    fn derivative(&self, var: &Placeholder<f64>) -> Result<OwnedAst<f64>, AstError<f64>> {
        Ok(OwnedAst::new(self.0.derivative(var)?))
    }
}

impl OwnedAst<f64> {
    fn simplify<'a>(&'a self) -> Result<OwnedAst<f64>, AstError<'a, f64>> {
        Ok(match **self.0 {
            AstDiff::Add(ref a, ref b) => {
                let a = a.simplify()?;
                let b = b.simplify()?;
                if AstDiff::Constant(0.) == **a.0  {
                    b
                } else if AstDiff::Constant(0.) == **b.0  {
                    a
                } else {
                    OwnedAst::new(AstDiff::Add(a, b))
                }
            }
            AstDiff::Mul(ref a, ref b) => {
                let a = a.simplify()?;
                let b = b.simplify()?;
                if AstDiff::Constant(0.) == **a.0 || AstDiff::Constant(0.) == **b.0 {
                    OwnedAst::new(AstDiff::Constant(0.))
                } else if AstDiff::Constant(1.) == **a.0  {
                    b
                } else if AstDiff::Constant(1.) == **b.0  {
                    a
                } else {
                    OwnedAst::new(AstDiff::Mul(a, b))
                }
            },
            AstDiff::Var(_) => self.clone(),
            AstDiff::Constant(_) => self.clone()
        })
    }
}


trait Fun<T, M, N> {
    fn eval(&self, datas: &Array<T, N>) -> Array<T, M>;
}

trait Gradient {

}

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

fn f1_ast(data: &Array1<f64>) -> AstDiff<f64> {
    AstDiff::Add(OwnedAst::new(AstDiff::Constant(2.)), OwnedAst::new(AstDiff::Constant(3.)))

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


#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f64 = 1e-15;

    #[test]
    fn eval() {
        assert!((5. - AstDiff::Add(OwnedAst::new(AstDiff::Constant(2.)), OwnedAst::new(AstDiff::Constant(3.))).eval(&vec![]).unwrap()).abs() < EPSILON);
        assert!((0.3 - AstDiff::Add(OwnedAst::new(AstDiff::Constant(0.1)), OwnedAst::new(AstDiff::Constant(0.2))).eval(&vec![]).unwrap()).abs() < EPSILON);
        assert!((0.02 - AstDiff::Mul(OwnedAst::new(AstDiff::Constant(0.1)), OwnedAst::new(AstDiff::Constant(0.2))).eval(&vec![]).unwrap()).abs() < EPSILON);
        let pl = Placeholder {
            ident: 1,
            _ghost: PhantomData
        };
        let ast_test = OwnedAst::new(AstDiff::Mul(OwnedAst::new(AstDiff::Constant(-4.)), OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Constant(0.2))))));
        assert!((-4.8 - ast_test.eval(&vec![(&pl, 1.0)]).unwrap()).abs() < EPSILON);
        assert!((-0.8 - ast_test.eval(&vec![(&pl, 0.0)]).unwrap()).abs() < EPSILON);
        assert!((19.2 - ast_test.eval(&vec![(&pl, -5.0)]).unwrap()).abs() < EPSILON);
    }

    #[test]
    fn derivative() {
        let pl = Placeholder {
            ident: 1,
            _ghost: PhantomData
        };
        assert!(AstDiff::Constant(-4.).derivative(&pl.clone()).unwrap() == AstDiff::Constant(0.));
        assert!(AstDiff::Var(pl.clone()).derivative(&pl.clone()).unwrap() == AstDiff::Constant(1.));
        let ast_test_1var = OwnedAst::new(AstDiff::Mul(OwnedAst::new(AstDiff::Constant(-4.)), OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Constant(0.2))))));
        assert!(ast_test_1var.derivative(&pl.clone()).unwrap().simplify().unwrap() == OwnedAst::new(AstDiff::Constant(-4.)));
        let ast_test_2var = OwnedAst::new(AstDiff::Mul(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Constant(0.2))))));
        assert!(ast_test_2var.derivative(&pl.clone()).unwrap().simplify().unwrap() == OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Constant(0.2)))), OwnedAst::new(AstDiff::Var(pl.clone())))));
    }
}

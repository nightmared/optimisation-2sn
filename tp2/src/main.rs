use ndarray::{Array, Array1, Array2, Dimension};
use ndarray_linalg::Norm;

use ndarray::prelude::*;
use ndarray_linalg::Solve;

use std::marker::PhantomData;
use std::sync::Arc;
use std::fmt::{Display, Formatter};
use std::collections::HashSet;

#[derive(Debug)]
struct Placeholder<T> {
    ident: u32,
    _ghost: PhantomData<T>
}

impl<T> std::hash::Hash for Placeholder<T> {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        self.ident.hash(hasher);
    }
}

impl<T: Display> Display for Placeholder<T> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "x_{}", self.ident)
    }
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

impl<T> Eq for Placeholder<T> {}

impl<T> PartialOrd for Placeholder<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.ident == other.ident {
            Some(std::cmp::Ordering::Equal)
        } else if self.ident < other.ident {
            Some(std::cmp::Ordering::Less)
        } else {
            Some(std::cmp::Ordering::Greater)
        }
    }
}

impl<T> Ord for Placeholder<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}


#[derive(Debug)]
enum AstError<T> {
    UnavailableVariable(Box<Placeholder<T>>)
}

trait Ast<T> {
    fn eval(&self, vars: &Vec<(&Placeholder<T>, f64)>) -> Result<T, AstError<T>>;
    fn derivative(&self, var: &Placeholder<T>) -> Result<Self, AstError<T>> where Self: Sized;
    fn get_vars(&self) -> HashSet<Placeholder<T>>;
}


#[derive(PartialEq, Debug)]
enum AstDiff<T> {
    Mul(OwnedAst<T>, OwnedAst<T>),
    Add(OwnedAst<T>, OwnedAst<T>),
    Var(Placeholder<T>),
    Constant(T)
}

impl<T: Display> Display for AstDiff<T> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        Ok(match self {
            AstDiff::Constant(c) => write!(f, "{}", c)?,
            AstDiff::Var(v) => write!(f, "{}", v)?,
            AstDiff::Mul(a, b) => write!(f, "({}*{})", a, b)?,
            AstDiff::Add(a, b) => write!(f, "({}+{})", a, b)?
        })
    }
}

#[derive(Clone, Debug)]
struct OwnedAst<T> (Arc<Box<AstDiff<T>>>);

impl<T: Display> Display for OwnedAst<T> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", **self.0)
    }
}

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
                Err(AstError::UnavailableVariable(Box::new(v.clone())))
            }
        }
    }
    fn derivative(&self, var: &Placeholder<f64>) -> Result<AstDiff<f64>, AstError<f64>> {
        match self {
            AstDiff::Constant(_) => Ok(AstDiff::Constant(0.)),
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
    fn get_vars(&self) -> HashSet<Placeholder<f64>> {
        match self {
            AstDiff::Constant(_) => HashSet::new(),
            AstDiff::Var(v) => {
                let mut set = HashSet::new();
                set.insert(v.clone());
                set
            },
            AstDiff::Add(a, b) => {
                a.get_vars().union(&b.get_vars()).into_iter().map(|x| x.clone()).collect()
            },
            AstDiff::Mul(a, b) => {
                a.get_vars().union(&b.get_vars()).into_iter().map(|x| x.clone()).collect()
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
    fn get_vars(&self) -> HashSet<Placeholder<f64>> {
        self.0.get_vars()
    }
}

impl OwnedAst<f64> {
    fn simplify(&self) -> Result<OwnedAst<f64>, AstError<f64>> {
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
    fn eval(&self, data: &Array<T, N>) -> Result<Array<OwnedAst<T>, M>, AstError<T>>;
    //fn grad(&self) -> Result<Gradient<T, M, N>, AstError<T>>;
}

struct Function<T, M, N> {
    fun: Array<OwnedAst<T>, N>,
    _out_ghost: PhantomData<M>
}

#[derive(Debug)]
struct Gradient<T> {
    fun: Array1<OwnedAst<T>>
}

impl<T: Display> Display for Gradient<T> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "(∇={})", self.fun)
    }
}

impl Gradient<f64> {
    fn gradient(fun: OwnedAst<f64>, vars: &Vec<Placeholder<f64>>) -> Result<Gradient<f64>, AstError<f64>> {
        let mut res = Vec::with_capacity(vars.len());
        for i in 1..=vars.len() {
            res.push(fun.derivative(&vars[i-1])?.simplify()?)
        }
        Ok(Gradient {
            fun: res.into()
        })
    }
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
    k_max: usize
}

fn newton(gradient: impl Fn(&Array1<f64>) -> Array1<f64>, hessienne: impl Fn(&Array1<f64>) -> Array2<f64>, x0: &Array1<f64>, params: &AlgoParams) -> (Array1<f64>, ExitCondition) {
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

fn f1(x1: f64, x2: f64, x3: f64) -> f64 {
    2.*(x1+x2+x3-3.).powi(2)+(x1-x2).powi(2)+(x2-x3).powi(2)
}

fn f1_ast(data: &Array1<f64>) -> OwnedAst<f64> {
    let mut v = vec![];
    for i in 1..=data.len() {
        v.push(Placeholder {
            ident: i as u32,
            _ghost: PhantomData
        });
    }
    OwnedAst::new(AstDiff::Mul(OwnedAst::new(AstDiff::Var(v[0].clone())), OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(v[1].clone())), OwnedAst::new(AstDiff::Constant(0.2))))))
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
        k_max: 500
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

    const pl: Placeholder<f64> = Placeholder {
        ident: 1,
        _ghost: PhantomData
    };
    const pl2: Placeholder<f64> = Placeholder {
        ident: 2,
        _ghost: PhantomData
    };

    #[test]
    fn eval() {
        assert!((5. - AstDiff::Add(OwnedAst::new(AstDiff::Constant(2.)), OwnedAst::new(AstDiff::Constant(3.))).eval(&vec![]).unwrap()).abs() < EPSILON);
        assert!((0.3 - AstDiff::Add(OwnedAst::new(AstDiff::Constant(0.1)), OwnedAst::new(AstDiff::Constant(0.2))).eval(&vec![]).unwrap()).abs() < EPSILON);
        assert!((0.02 - AstDiff::Mul(OwnedAst::new(AstDiff::Constant(0.1)), OwnedAst::new(AstDiff::Constant(0.2))).eval(&vec![]).unwrap()).abs() < EPSILON);

        let ast_test = OwnedAst::new(AstDiff::Mul(OwnedAst::new(AstDiff::Constant(-4.)), OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Constant(0.2))))));
        assert!((-4.8 - ast_test.eval(&vec![(&pl, 1.0)]).unwrap()).abs() < EPSILON);
        assert!((-0.8 - ast_test.eval(&vec![(&pl, 0.0)]).unwrap()).abs() < EPSILON);
        assert!((19.2 - ast_test.eval(&vec![(&pl, -5.0)]).unwrap()).abs() < EPSILON);
    }

    #[test]
    fn derivative() {
        assert!(AstDiff::Constant(-4.).derivative(&pl.clone()).unwrap() == AstDiff::Constant(0.));
        assert!(AstDiff::Var(pl.clone()).derivative(&pl.clone()).unwrap() == AstDiff::Constant(1.));

        let ast_test_1_var = OwnedAst::new(AstDiff::Mul(OwnedAst::new(AstDiff::Constant(-4.)), OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Constant(0.2))))));
        assert!(ast_test_1_var.derivative(&pl.clone()).unwrap().simplify().unwrap() == OwnedAst::new(AstDiff::Constant(-4.)));

        // (x_1*(x_1+0.2)) -> ((x_1+0.2)+x_1)
        let ast_test_2_id_var = OwnedAst::new(AstDiff::Mul(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Constant(0.2))))));
        assert!(ast_test_2_id_var.derivative(&pl.clone()).unwrap().simplify().unwrap() == OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Constant(0.2)))), OwnedAst::new(AstDiff::Var(pl.clone())))));

        // (x_1*(x_2+0.2)) -> (x_2+0.2) and x_1
        let ast_test_2_var = OwnedAst::new(AstDiff::Mul(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl2.clone())), OwnedAst::new(AstDiff::Constant(0.2))))));
        assert!(ast_test_2_var.derivative(&pl.clone()).unwrap().simplify().unwrap() == OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl2.clone())), OwnedAst::new(AstDiff::Constant(0.2)))));
        assert!(ast_test_2_var.derivative(&pl2.clone()).unwrap().simplify().unwrap() == OwnedAst::new(AstDiff::Var(pl.clone())));
    }

    #[test]
    fn gradient() {
        // (x_1*(x_2+0.2)) -> ((x_2+0.2), x_1)^T
        let ast_test_2_var = OwnedAst::new(AstDiff::Mul(OwnedAst::new(AstDiff::Var(pl.clone())), OwnedAst::new(AstDiff::Add(OwnedAst::new(AstDiff::Var(pl2.clone())), OwnedAst::new(AstDiff::Constant(0.2))))));

        println!("{}", Gradient::gradient(ast_test_2_var, &vec![pl.clone(), pl2.clone()]).unwrap());
    }
}

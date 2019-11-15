use std::marker::PhantomData;
use crate::Placeholder;

#[derive(Debug, Clone, PartialEq)]
enum ParserError<E> {
    InvalidExpr(String),
    FromStrErr(E)

}

#[derive(Debug, Clone, PartialEq)]
enum Token<T> {
    Plus,
    Sub,
    Times,
    Over,
    ParLeft,
    ParRight,
    Var(Placeholder<T>),
    Num(T)
}

fn is_separator_char(s: char) -> bool {
    s.is_whitespace() || s == '(' || s == ')'
}

fn match_word<T: std::str::FromStr>(s_bytes: &[u8]) -> Result<Token<T>, ParserError<T::Err>> {
    let s_str = std::str::from_utf8(s_bytes).unwrap();
    Ok(match s_str {
            "+" => Token::Plus,
            "-" => Token::Sub,
            "*" => Token::Times,
            "/" => Token::Over,
            "(" => Token::ParLeft,
            ")" => Token::ParRight,
            _ => {
                // is it a variable
                if b'x' == s_bytes[0] {
                    if let Ok(idx) = std::str::from_utf8(&s_bytes[1..]).unwrap().parse() as Result<u32, std::num::ParseIntError> {
                        Token::Var(Placeholder {
                            ident: idx,
                            _ghost: PhantomData
                        })
                    } else {
                        return Err(ParserError::InvalidExpr(s_str.to_string()));
                    }
                // try to match a number
                } else if let Ok(x) = s_str.parse() {
                    Token::Num(x)
                } else {
                    return Err(ParserError::InvalidExpr(s_str.to_string()));
                }
            }
        })
}

fn tokenize<T: std::str::FromStr>(s: &str) -> Result<Vec<Token<T>>, ParserError<T::Err>> {
    let s: Vec<u8> = s.bytes().collect();

    let mut v = Vec::new();

    let mut start = 0;
    for i in 0..s.len() {
        if s[i] == b' ' {
            if i == start {
                start += 1;
            } else {
                let word = match_word(&s[start..i])?;
                v.push(word);
                start = i + 1;
            }
            continue;
        } else if s[i] == b'(' || s[i] == b')' {
            v.push(match_word(&s[i..i+1])?);
            start += 1;
        }
    }
    // leftover ?
    if start != s.len() {
        v.push(match_word(&s[start..s.len()])?);
        

    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    const x1: Placeholder<f64> = Placeholder {
        ident: 1,
        _ghost: PhantomData
    };
    const x2: Placeholder<f64> = Placeholder {
        ident: 2,
        _ghost: PhantomData
    };


    #[test]
    fn tokenizer() {
        assert_eq!(tokenize("1 + 2.5").unwrap(), vec![Token::Num(1.), Token::Plus, Token::Num(2.5)]);
        assert_eq!(tokenize("1 + (2.5 * x1) / x2").unwrap(), vec![Token::Num(1.), Token::Plus, Token::ParLeft, Token::Num(2.5), Token::Times, Token::Var(x1.clone()), Token::ParRight, Token::Over, Token::Var(x2.clone())]);

    }
}

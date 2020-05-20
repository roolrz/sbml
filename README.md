# sbml
Homework for CSE 307@Stony Brook U<br />

Build own Programming Language using Python and ply package<br />

# Usage
./sbml.py "path to program" (in *nix environment)<br />
or<br />
python3 sbml.py "path to program"<br />

# Program Syntax

## Variables
Variables must be fit in the regure expression '[a-zA-Z][a-zA-Z0-9_]' and should not be any keyword pre-defined in the program

## Expression
Expressions are those can be evaluated value. Function calls, Variables with value assigned, list, tuple, list indexing, tuple indexing... These can be expressions

## Statement
Statement don't have a value. If you assign a statement to a variable, you'll got undefined result.<br />

The syntax of statement is assign_statement, print_statement, if_statement, if_else_statement or while_loop_statement. There must be a semicolon ';' at the end of print statement or assignment.

## Data tpyes
The data type accept list, tuple, float, int, boolean. It's basically share the same syntax to the Python, only the basic usage of python. Usage like list[1:-1] is not supported.

One thing need to noticed is that the indexing expression of tuple, we use #i(tuple_name) for the indexing operation. Also index is the *REAL* index, which means tuple_name[0] in python equals to #1(tuple_name) in sbml.

## Main function
Main function starts with '{' and end with '}'. Main function must be in the last part of the program file. Which means function definitions must before the final main function block.<br />
So there's no need to define the main function using main keyword.

## Function defination
A function definition begins with the keyword "fun", followed by the name of the function, a left parenthesis, variables representing formal parameters separated by commas, a right parenthesis, an equal sign, a block, and then an expression.

# Sample programs
filename: nofunction.sbml
```
{
  print("Hi!");
}
```

filename: justfunction.sbml
```
fun f() = 
{
  output = "Hello";
  print("Printed inside function f().");
}
output;
{
  x = f();
  print(x);
}
```

filename: bubblesort.sbml
```
fun bubblesort(alist, length) = 
{
  i = 0;
  while (i < (length -1))
  {
    j = 0;
    while (j < ((length - i) - 1))
    {
      next = (j + 1);
      if (alist[j] > alist[next])
      {
        temp = alist[j];
        alist[j] = alist[next];
        alist[next] = temp;
      }
      j = j + 1;
    }
    i = i + 1;
  }
  output = alist;
}
output;
{
  print("Sorted list is: [7, 19, 32, 44, 56, 89, 122, 330]");
  inlist = [122, 44, 32, 7, 89, 56, 330, 19];
  outlist = bubblesort(inlist, 8);
  n = 0;
  while (n < 8)
  {
    print(outlist[n]);
    n = n + 1;
  }
}
```
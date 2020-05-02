#! /usr/bin/env python3

from ply import lex
from ply import yacc
import sys

class SyntaxException(Exception):
        pass

class SemanticException(Exception):
        pass

class NameException(Exception):
        pass

class VariableBuilder:
        def __init__(self):
                self.table = {}

        def check(self, name):
                if not isinstance(name, str):
                        raise NameException
                if name in self.table:
                        return True
                else:
                        return False

        def delete(self, name):
                if not isinstance(name, str):
                        raise NameException
                if name in self.table:
                        self.table.pop(name, None)
                        return True
                else:
                        return False

        def read(self, name):
                if self.check(name):
                        return self.table[name]
                else:
                        return None

        def write(self, name, value):
                if self.check(name):
                        self.delete(name)
                        self.table[name] = value
                else:
                        self.table[name] = value
                return True
                
global varTable
varTable = VariableBuilder()

keywords = {
        'True'          : 'TRUE',
        'False'         : 'FALSE',
        'div'           : 'DIV',
        'mod'           : 'MOD',
        'in'            : 'IN',
        'not'           : 'NOT',
        'andalso'       : 'ANDALSO',
        'orelse'        : 'ORELSE',
        'print'         : 'PRINT',
        'if'            : 'IF',
        'else'          : 'ELSE',
        'while'         : 'WHILE', 
}

tokens = list(keywords.values()) + [
        'NAME', # reserved to process keywords
        'INTNUM',
        'REALNUM',
        'PLUS',
        'MINUS',
        'TIMES',
        'DIVIDE',
        'HASHTAG',
        'EXPONENT',
        'STRING',
        'LPAREN',
        'RPAREN',
        'LBRACKET',
        'RBRACKET',
        'LBLOCK',
        'RBLOCK',
        'COMMA',
        'CONS',
        'LT',
        'LEQ',
        'EQU',
        'NEQ',
        'GEQ',
        'GT',
        'ASSIGN',
        'SEMICOLON'
]

t_REALNUM = r'\d*\.\d+e\-\d+|\d*\.\d+e\d+|\d+\.\d*e\-\d+|\d+\.\d*e\d+|\d*\.\d+|\d+\.\d*'
t_INTNUM  = r'\d+'
t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_HASHTAG = r'\#'
t_EXPONENT= r'\*\*'

def t_STRING(t):
        r'\".*?\"|\'.*?\''
        t.value = t.value[1:-1]
        return t

t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_LBRACKET= r'\['
t_RBRACKET= r'\]'
t_LBLOCK  = r'\{'
t_RBLOCK  = r'\}'
t_COMMA   = r','
t_CONS    = r'::'
t_LT      = r'<'
t_LEQ     = r'<='
t_EQU     = r'=='
t_NEQ     = r'<>'
t_GEQ     = r'>='
t_GT      = r'>'
t_ASSIGN  = r'='
t_SEMICOLON = ';'

def t_NAME(t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = keywords.get(t.value,'NAME')
        return t

def t_newline(t):
        r'\n+'
        t.lexer.lineno += len(t.value)

t_ignore  = ' \t'

def t_error(t):
        raise SyntaxException
        t.lexer.skip(1)

# Parser start here
class Statement:
        def __init__(self):
                pass
        def execute(self):
                return 0

class Block(Statement):
        def __init__(self, statements):
                self.statements = statements

        def execute(self):
                for statement in self.statements:
                        statement.execute()

class PrintStatement(Statement):
        def __init__(self, obj):
                self.obj = obj
        
        def execute(self):
                V = self.obj.evaluate()
                print(V)

class AssignStatement(Statement):
        def __init__(self, name, val):
                self.name = name
                self.val = val

        def execute(self):
                V = self.val.evaluate()
                varTable.write(self.name, V)

class AssignListElemStatement(Statement):
        def __init__(self, name, idx, value):
                self.name = name
                self.idx = idx
                self.value = value
        
        def execute(self):
                N = varTable.read(self.name)
                V = self.value.evaluate()
                I = self.idx.evaluate()
                if not isinstance(N, list):
                        raise SemanticException
                if not isinstance(I, list):
                        raise SemanticException
                if len(I) != 1:
                        raise SemanticException
                I = I[0]
                if I >= len(N) or I < 0 or (not isinstance(I, int)):
                        raise SemanticException
                N[I] = V
                varTable.write(self.name, N)

class IfStatement(Statement):
        def __init__(self, condition, blk):
                self.condition = condition
                self.blk = blk
        
        def execute(self):
                if  self.condition.evaluate() == True:
                        self.blk.execute()

class IfElseStatement(Statement):
        def __init__(self, condition, blk, else_blk):
                self.condition = condition
                self.blk = blk
                self.else_blk = else_blk
        
        def execute(self):
                if self.condition.evaluate() == True:
                        self.blk.execute()
                else:
                        self.else_blk.execute()

class WhileLoopStatement(Statement):
        def __init__(self, condition, blk):
                self.condition = condition
                self.blk = blk
        
        def execute(self):
                while self.condition.evaluate() != False:
                        self.blk.execute()
                

class Expr:
        def __init__(self):
                pass
        def evaluate(self):
                return 0
        def execute(self):
                pass

class NameExpr(Expr):
        def __init__(self, name):
                self.name = name
        
        def evaluate(self):
                ret = varTable.read(self.name)
                if ret == None:
                        raise NameException
                else:
                        return ret

class UminusNumExpr(Expr):
        def __init__(self, value):
                self.value = value

        def evaluate(self):
                V = self.value.evaluate()
                if isinstance(V, (int, float)):
                        return -V
                else:
                        raise SemanticException

class NumExpr(Expr):
        def __init__(self, value):
                self.value = value

        def evaluate(self):
                if isinstance(self.value, str):
                        try:
                                V = int(self.value)
                        except ValueError:
                                V = float(self.value)
                        return V
                else:
                        return self.value

class StringExpr(Expr):
        def __init__(self, value):
                self.value = value
        
        def evaluate(self):
                return self.value

class BoolExpr(Expr):
        def __init__(self, value):
                self.value = value
        
        def evaluate(self):
                if self.value == 'True':
                        return True
                elif self.value == 'False':
                        return False
                else:
                        return self.value

class TupleExpr(Expr):
        def __init__(self, value, value_extra=None):
                self.value = value
                self.value_extra = value_extra

        def evaluate(self):
                if self.value_extra == None:
                        return (self.value.evaluate(),)
                V = self.value.evaluate()
                VE = self.value_extra.evaluate()
                return (V,) + VE

class EmptyListExpr(Expr):
        def __init__(self):
                self.empty = 1
        
        def evaluate(self):
                return []

class ListExpr(Expr):
        def __init__(self, value, following=None):
                self.value = value
                self.following = following

        def evaluate(self):
                if self.following == None:
                        return [self.value.evaluate()]
                V = self.value.evaluate()
                return [V]+self.following.evaluate()


class ParenExpr(Expr):
        def __init__(self, value):
                self.value = value
        
        def evaluate(self):
                return self.value.evaluate()

class LOpSingleExpr(Expr):
        def __init__(self, op, value):
                self.op = op
                self.value = value

        def evaluate(self):
                V = self.value.evaluate()

                if self.op == 'not':
                        if isinstance(V, bool):
                                return not V
                raise SemanticException

class MidOpDoubleExpr(Expr):
        def __init__(self, op, lvalue, rvalue):
                self.lvalue = lvalue
                self.rvalue = rvalue
                self.op = op

        def evaluate(self):
                L = self.lvalue.evaluate()
                R = self.rvalue.evaluate()

                if (isinstance(L, bool) and isinstance(R, bool)):
                        if self.op == 'orelse':
                                return (L or R)
                        elif self.op == 'andalso':
                                return (L and R)
                        else:
                                raise SemanticException
                elif (isinstance(L, (int, float)) and isinstance(R, (int, float))):
                        if self.op == '<':
                                return (L < R)
                        elif self.op == '<=':
                                return (L <= R)
                        elif self.op == '==':
                                return (L == R)
                        elif self.op == '<>':
                                return (L != R)
                        elif self.op == '>=':
                                return (L >= R)
                        elif self.op == '>':
                                return (L > R)
                        elif self.op == '+':
                                return (L + R)
                        elif self.op == '-':
                                return (L - R)
                        elif self.op == '*':
                                return (L * R)
                        elif self.op == '/':
                                if R == 0:
                                        raise SemanticException

                                return float(L / R)
                        elif self.op == 'div':
                                if R == 0:
                                        raise SemanticException

                                if (isinstance(L, int) and isinstance(R, int)):
                                        return (L // R)
                                else:
                                        raise SemanticException
                        elif self.op == 'mod':
                                if R == 0:
                                        raise SemanticException

                                if (isinstance(L, int) and isinstance(R, int)):
                                        return (L % R)
                                else:
                                        raise SemanticException
                        elif self.op == '**':
                                return (L ** R)
                        else:
                                raise SemanticException
                elif (isinstance(L, str) and isinstance(R, str)):
                        if self.op == '<':
                                return (L < R)
                        elif self.op == '<=':
                                return (L <= R)
                        elif self.op == '==':
                                return (L == R)
                        elif self.op == '<>':
                                return (L != R)
                        elif self.op == '>=':
                                return (L >= R)
                        elif self.op == '>':
                                return (L > R)
                        elif self.op == '+':
                                return (L + R)
                        elif self.op == 'in':
                                return (L in R)
                        else:
                                raise SemanticException
                elif (isinstance(L, list) and isinstance(R, list)):
                        if self.op == '+':
                                return (L + R)
                elif self.op == '::':
                        if isinstance(R, list):
                                return ([L] + R)
                        else:
                                raise SemanticException
                elif self.op == 'in':
                        if isinstance(R, list):
                                return (L in R)
                        elif (isinstance(L, str) and isinstance(R, str)):
                                return (L in R)
                        else:
                                raise SemanticException
                else:
                        raise SemanticException


class IndexingExpr(Expr):
        def __init__(self, obj, idx):
                self.obj = obj
                self.idx = idx
        
        def evaluate(self):
                O = self.obj.evaluate()
                I = self.idx.evaluate()
                if not isinstance(O, (str, list)):
                        raise SemanticException
                if not isinstance(I, list):
                        raise SemanticException
                if len(I) != 1:
                        raise SemanticException
                I = I[0]
                if I >= len(O) or I < 0 or isinstance(I, bool):
                        raise SemanticException
                return O[I]


class TupleIndexingExpr(Expr):
        def __init__(self, idx, tup):
                self.idx = idx
                self.tup = tup
        
        def evaluate(self):
                T = self.tup.evaluate()
                I = self.idx.evaluate()
                I = I - 1
                if not isinstance(T, tuple):
                        raise SemanticException
                if not isinstance(I, int):
                        raise SemanticException
                if I >= len(T) or I < 0:
                        raise SemanticException
                return T[I]

precedence = (
        ('left', 'ASSIGN'),
        ('left', 'ORELSE'),
        ('left', 'ANDALSO'),
        ('left', 'NOT'),
        ('left', 'LT', 'LEQ', 'EQU', 'NEQ', 'GEQ', 'GT'),
        ('right', 'CONS'),
        ('left', 'IN'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE', 'DIV', 'MOD'),
        ('right','UMINUS'),
        ('right', 'EXPONENT'),
        ('left', 'LBRACKET', 'RBRACKET'),
        ('left', 'HASHTAG'),
        ('left', 'COMMA'),
        ('left', 'LPAREN', 'RPAREN'),
)

def p_program(p):
        ' program : block '
        p[0] = p[1] 

def p_blocks(p):
        ' block : LBLOCK statements RBLOCK '
        p[0] = Block(p[2])

def p_blocks_empty(p):
        ' block : LBLOCK RBLOCK '
        p[0] = Block([])

def p_statements(p):
        ''' statements : statement statements 
                        | statement '''
        try:
                p[0] = [p[1]]+ p[2]
        except IndexError:
                p[0] = [p[1]]

def p_statement(p):
        ''' statement : assign_statement
                        | print_statement  
                        | if_statement 
                        | if_else_statement 
                        | while_loop_statement '''
        p[0] = p[1]

def p_assign_statement(p):
        ''' assign_statement : NAME ASSIGN expression SEMICOLON'''
        p[0] = AssignStatement(p[1], p[3])

def p_assign_statement_list(p):
        ''' assign_statement : NAME list_expr ASSIGN expression SEMICOLON'''
        p[0] = AssignListElemStatement(p[1], p[2], p[4])

def p_print_statement(p):
        ' print_statement : PRINT LPAREN expression RPAREN SEMICOLON'
        p[0] = PrintStatement(p[3])

def p_if_statement(p):
        ' if_statement : IF LPAREN expression RPAREN block'
        p[0] = IfStatement(p[3], p[5])

def p_if_else_statement(p):
        ' if_else_statement : IF LPAREN expression RPAREN block ELSE block'
        p[0] = IfElseStatement(p[3], p[5], p[7])

def p_while_loop_statement(p):
        ' while_loop_statement : WHILE LPAREN expression RPAREN block '
        p[0] = WhileLoopStatement(p[3], p[5])

def p_expression(p):
        ''' expression : number_expr
                        | name_expr
                        | uminus_expr
                        | boolean_expr
                        | string_expr
                        | tuple_expr
                        | list_expr
                        | paren_expr
                        | lop_single_expr 
                        | midop_double_expr 
                        | indexing_expr
                        | tuple_indexing_expr '''
        p[0] = p[1]

def p_name_expr(p):
        ' name_expr : NAME '
        p[0] = NameExpr(p[1])

def p_uminus_expr(p):
        'uminus_expr : MINUS expression %prec UMINUS'
        p[0] = UminusNumExpr(p[2])

def p_number_expr(p):
        ''' number_expr : INTNUM
                        | REALNUM '''
        p[0] = NumExpr(p[1])

def p_boolean_expr(p):
        ''' boolean_expr : TRUE
                        | FALSE '''
        p[0] = BoolExpr(p[1])

def p_string_expr(p):
        ' string_expr : STRING'
        p[0] = StringExpr(p[1])

def p_tuple_expr(p):
        ''' tuple_expr : LPAREN p_tuple_expr_tail '''
        p[0] = p[2]

def p_tuple_expr_tail(p):
        ''' p_tuple_expr_tail : expression COMMA p_tuple_expr_tail
                        | expression COMMA expression RPAREN
                        | expression COMMA RPAREN '''
        if p[3] == ')':
                p[0] = TupleExpr(p[1])
        elif p[3] != ')' and len(p) > 4:
                p[0] = TupleExpr(p[1], TupleExpr(p[3]))
        else:
                p[0] = TupleExpr(p[1], p[3])

def p_list_expr(p):
        ''' list_expr : LBRACKET list_expr_body RBRACKET 
                        | LBRACKET RBRACKET'''
        if p[2] == ']':
                p[0] = EmptyListExpr()
        else:
                p[0] = p[2]

def p_list_expr_body(p):
        ''' list_expr_body : expression COMMA list_expr_body
                        | expression 
                        | expression COMMA'''
        try:
                p[0] = ListExpr(p[1], following=p[3])
        except IndexError:
                p[0] = ListExpr(p[1])

def p_paren_expr(p):
        ' paren_expr : LPAREN expression RPAREN'
        p[0] = ParenExpr(p[2])

def p_lop_single_expr(p):
        ' lop_single_expr : NOT expression '
        p[0] = LOpSingleExpr(p[1], p[2])

def p_midop_double_expr(p):
        ''' midop_double_expr : expression ORELSE expression 
                        | expression ANDALSO expression
                        | expression LT expression
                        | expression LEQ expression
                        | expression EQU expression
                        | expression NEQ expression
                        | expression GEQ expression
                        | expression GT expression
                        | expression PLUS expression
                        | expression MINUS expression
                        | expression TIMES expression
                        | expression DIVIDE expression
                        | expression DIV expression
                        | expression MOD expression
                        | expression EXPONENT expression 
                        | expression IN expression
                        | expression CONS expression'''
        p[0] = MidOpDoubleExpr(p[2], p[1], p[3])

def p_indexing_expr(p):
        'indexing_expr : expression list_expr'
        p[0] = IndexingExpr(p[1], p[2])

def p_tuple_indexing_expr(p):
        'tuple_indexing_expr : HASHTAG number_expr expression'
        p[0] = TupleIndexingExpr(p[2], p[3])

def p_error(p):
        #print("Semantic error at '%s'" % p)
        raise SyntaxException
        

if __name__ == '__main__':
        debugmode = False
        lexer = lex.lex()
        parser = yacc.yacc()

        try:
                file = open(sys.argv[1],"r")
        except FileNotFoundError:
                print('Unable to open file "' + str(sys.argv[1]) + '"')
                exit()
        except IndexError:
                debugmode = True 

        if debugmode:
                while True:
                        try:
                                expr = input()
                        except EOFError:
                                print("Received Ctrl+D, Exiting...")
                                exit()

                        try:
                                lexer.input(expr)
                                for token in lexer:
                                        print(token)
                                parser.parse(expr).execute()
                        except SemanticException:
                                print('SEMANTIC ERROR')
                        except SyntaxException:
                                print('SYNTAX ERROR')
                        except NameException:
                                print('UNDEFINED VARIABLE')
        else:
                exprs = file.read()
                exprs = ''.join(exprs)
                try:
                        parser.parse(exprs).execute()
                except SemanticException:
                        print('SEMANTIC ERROR')
                except SyntaxException:
                        print('SYNTAX ERROR')
                except NameException:
                        print('SYNTAX ERROR')

        file.close()
                
                

import textwrap
DOMAIN_MAP = {0:'general', 1:'code', 2:'legal',3:'customer_service'}
ROUTER_TEMPLATE =   '''\
    User Question: {query}
    Command = Please output only the classification number, must be one of 0, 1, 2, 3, where 0=general, 1=code, 2=legal, 3=customer_service. Do not output any extra text.
    label:'''
ROUTER_PROMPT_TEMPLATE = textwrap.dedent(ROUTER_TEMPLATE)

ANSWER_TEMPLATE = '''\
    User Question: {query}\nAnswer:'''
ANSWER_PROMPT_TEMPLATE = textwrap.dedent(ANSWER_TEMPLATE)

COMPRESS_TEMPLATE = '''\
    Compress the following question from user into core information within
    one to two sentences. Keep all details and constraints, and only
    output the question compressed.
    
    Question: {query}
    Compress:'''
COMPRESS_PROMPT_TEMPLATE = textwrap.dedent(COMPRESS_TEMPLATE)

if __name__ == '__main__':
    print(repr(ROUTER_PROMPT_TEMPLATE.format(query='test')))

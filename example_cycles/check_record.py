from openmdao.recorders.case_reader import CaseReader

cr = CaseReader('propulsor.db')

print(cr.system_cases.list_cases())
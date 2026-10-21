"""Complete Ch3 workflow, sequential high-memory simulations (~several minutes)."""
import ch3_linescan as m
from patent_ringing import figures
import ch3_report
import importlib
from ideal_na06 import save_json


def main():
    c=m.config();o=m.optimize(c)
    figures()
    for case,choice in [("dots","none"),("dots","selected"),("dots","background_only"),("dots","best_hard"),
                        ("stripes","none"),("stripes","selected")]:
        m.full_simulation(c,o[choice],case,tag=choice)
    for case in ["dots","stripes"]:
        for choice in ["none","selected"]:
            m.full_simulation(c,o[choice],case,n=8192,window=1.024,tag=choice+"_fine")
    m.full_simulation(c,o["selected"],"dots",n=8192,window=2.048,tag="selected_wide")
    m.full_simulation(c,o["selected"],"dots",order=5,tag="selected_q5")
    design=m.design_parameters(o["selected"])
    result=m.evaluate_candidate(c,m.search_arrays(c,65536,4.096),{**design,"label":"fine_search"},details=True)
    save_json(m.OUT/"fine_search_validation.json",result)
    m.export_rods(c,o["selected"])
    importlib.reload(ch3_report).main()


if __name__=="__main__":main()

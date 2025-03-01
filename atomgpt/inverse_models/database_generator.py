from jarvis.core.specie import atomic_numbers_to_symbols
import numpy as np
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.composition import Composition
from atomgpt.inverse_models.inverse_models import (
    load_model,
    get_input,
    batch_evaluate,
)
import time
from itertools import combinations_with_replacement, permutations
from jarvis.core.atoms import Atoms


class AtomicDBGenerator:
    def __init__(
        self,
        max_atomic_number=100,
        max_stoichiometry=2,
        elements=None,
        model_path="",
        config=None,
        tokenizer=None,
        model=None,
        target=10,
        batch_size=2,
    ):
        self.max_atomic_number = max_atomic_number
        self.max_stoichiometry = (
            max_stoichiometry  # Maximum number of elements in a compound
        )
        self.model_path = model_path
        self.elements = elements or []
        self.target = str(target)
        self.batch_size = batch_size
        if not self.elements:
            Z = np.arange(max_atomic_number) + 1
            self.elements = atomic_numbers_to_symbols(Z)
        self.elements = list(set(self.elements))
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        if self.model_path == "" and self.model is None:
            raise ValueError("Provide model_path")
        if self.model is None:
            model, tokenizer, config = load_model(path=self.model_path)
            self.model = model
            self.tokenizer = tokenizer
            self.config = config

    def generate_samples(self):
        t1 = time.time()
        mem = {}
        inputs = set()  # Use a set to ensure uniqueness
        compositions = set()  # To keep track of unique compositions

        for stoich_count in range(
            1, self.max_stoichiometry + 1
        ):  # From unary to desired stoichiometry
            for comb in combinations_with_replacement(
                self.elements, stoich_count
            ):
                for perm in permutations(comb):  # To generate all orderings
                    try:
                        comp_dict = {el: perm.count(el) for el in set(perm)}
                        comp = Composition.from_dict(comp_dict)
                        reduced_formula = comp.reduced_formula

                        if reduced_formula not in compositions:
                            compositions.add(reduced_formula)
                            inp = get_input(
                                config=self.config,
                                chem=reduced_formula,
                                val=self.target,
                            )
                            inputs.add(
                                inp
                            )  # Add to inputs to ensure uniqueness
                    except Exception as exp:
                        print("Exp", exp)
                        pass

        mem["inputs"] = list(inputs)
        mem["outputs"] = batch_evaluate(
            prompts=list(inputs),
            model=self.model,
            tokenizer=self.tokenizer,
            csv_out="out.csv",
            config=self.config,
            batch_size=self.batch_size,
        )
        # for i,j in mem.items():
        #  print(i,j)
        fname = f"materials_stoichiometry_{self.max_stoichiometry}.json"
        t2 = time.time()
        mem["time"] = t2 - t1
        dumpjson(data=mem, filename=fname)
        print(f"Time taken for up to {self.max_stoichiometry}-ary: {t2 - t1}")
        return mem


if __name__ == "__main__":
    gen = AtomicDBGenerator(
        elements=["Mg", "B", "C"],
        max_stoichiometry=2,  # Can be set to any desired order
        model_path="/wrk/knc6/Software/atomgpt_opt/atomgpt/lora_model_m/",
        batch_size=10,
    )
    gen.generate_samples()

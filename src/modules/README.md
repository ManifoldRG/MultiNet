# GenESIS: Generalizable Extendable Stratified Inference System

MultiNet project uses the framework called **GenESIS**: **Gen**eralizable and **E**xtendable **S**tratified **I**nference **S**ystem to adapt a wide range of models into multiple types of tasks or datasets for scaling effectively while reducing the engineering efforts as much as possible. The core insights of GenESIS are 1) <u>Interchangeability</u>: Any models or datasets should interchangeably support each other, 2) <u>Abstraction</u>: Each module should share the same architecture and logic, and 3) <u>Encapsulation</u>: The programmer does not have to know the details on other modules and is allowed to concentrate on the models or datasets that are targeted. In this way, any models or datasets can easily be added to the MultiNet benchmark without affecting the existing implementations.

<img src="../../assets/framework-figure.png" alt="The figure of the inference framework."/>

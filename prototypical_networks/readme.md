

### Install ###

I recommend using a Python virtual environment
```bash
nlp_project:$ git clone https://github.com/clinc/oos-eval
nlp_project:$ pip install -r requirements.txt
nlp_project:$ python3 train_prototypical_bert.py
nlp_project:$ python3 build_prototypes.py
nlp_project:$ python3 test_noise_robustness.py
```

If you wish to use interactive mode run this: 
```bash
nlp_project:$ python3 build_prototypes.py --interactive
```
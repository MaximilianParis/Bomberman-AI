

Hier wird eine AI für Bomberman trainiert. Diese wird erst supervised trainiert, um wie ein regelbasierter Experte zu spielen. Anschließend wird die supervised AI durch Reinforcement-Learning mittels V-Trace-Target trainiert, den eigenen Reward direkt zu steigern.

Das Training findet hier statt: Bomberman-AI/bomberman-drl/scripts/

Der Experte ist in Expert_Agent.py implementiert. Das Supervised-Policy-Netz wird in Train_Supervised.py trainiert. Die Value Function wird in Train_Value_Function_Supervised.py erstellt. Zusammen werden diese weiter in Single_Worker_V-Trace.py optmiert.


from astar import graph_search, manhattan, zero_heuristic
from grid import create_experiment_grids
from markov import get_policy, build_transition_matrix, simulate_monte_carlo, analyze_absorption, compute_pi_n, build_transition_graph, identify_classes, check_periodicity
import matplotlib.pyplot as plt

def run_experiment_E1():
    print("\n" + "="*60)
    print("=== E.1 : Comparaison UCS / Greedy / A* sur 3 grilles ===")
    print("="*60)
    
    grilles = create_experiment_grids()
    for nom, env in grilles.items():
        print(f"\n>> GRILLE : {nom.upper()}")
        # UCS : w_g=1, w_h=0
        _, c_ucs, n_ucs, t_ucs, m_ucs = graph_search(env, zero_heuristic, 1, 0)
        # Greedy : w_g=0, w_h=1
        _, c_greedy, n_greedy, t_greedy, m_greedy = graph_search(env, manhattan, 0, 1)
        # A* : w_g=1, w_h=1
        _, c_astar, n_astar, t_astar, m_astar = graph_search(env, manhattan, 1, 1)
        
        print("+---------+--------+---------+------------+----------+")
        print("| Algo    | Coût   | Nœuds   | Temps (s)  | Max OPEN |")
        print("+---------+--------+---------+------------+----------+")
        print(f"| UCS     | {c_ucs:>6} | {n_ucs:>7} | {t_ucs:>10.4f} | {m_ucs:>8} |")
        print(f"| Greedy  | {c_greedy:>6} | {n_greedy:>7} | {t_greedy:>10.4f} | {m_greedy:>8} |")
        print(f"| A* | {c_astar:>6} | {n_astar:>7} | {t_astar:>10.4f} | {m_astar:>8} |")
        print("+---------+--------+---------+------------+----------+")

def run_experiment_E2():
    print("\n\n" + "="*68)
    print("=== E.2 : Variation de l'incertitude ε (Grille Moyenne) ===")
    print("="*68)
    
    grilles = create_experiment_grids()
    env = grilles['moyenne']
    epsilons = [0.0, 0.1, 0.2, 0.3]
    
    # Calcul du coût A* (déterministe, indépendant de ε)
    _, cost_astar, _, _, _ = graph_search(env, manhattan, 1, 1)
    
    print("\n+------+-------------------+------------------+----------------------+------------+")
    print("|  ε   | Proba GOAL (théo) | Proba GOAL (sim) | Temps moyen (succès) | Coût A*    |")
    print("+------+-------------------+------------------+----------------------+------------+")
    
    for eps in epsilons:
        policy = get_policy(env, env.goal)
        P, states, _ = build_transition_matrix(env, policy, eps)
        
        # Analyse théorique (Markov)
        prob_theo, _ = analyze_absorption(P, states, env.start, env.goal)
        # Simulation (Monte-Carlo)
        prob_sim, t_suc_sim, _ = simulate_monte_carlo(env, P, states, env.start, env.goal)
        
        print(f"| {eps:>4.1f} | {prob_theo:>17.4f} | {prob_sim:>16.4f} | {t_suc_sim:>20.2f} | {cost_astar:>10} |")
        
    print("+------+-------------------+------------------+----------------------+------------+")

def run_experiment_E3():
    print("\n\n" + "="*60)
    print("=== E.3 : Comparaison des heuristiques admissibles ===")
    print("===       (Grille Difficile, A* standard w_g=1)    ===")
    print("="*60)
    
    grilles = create_experiment_grids()
    env = grilles['difficile']
    
    # Test avec heuristique nulle (h=0) => équivalent à Dijkstra/UCS
    _, c_h0, n_h0, t_h0, _ = graph_search(env, zero_heuristic, 1, 1)
    # Test avec heuristique de Manhattan
    _, c_manh, n_manh, t_manh, _ = graph_search(env, manhattan, 1, 1)

    print("\n+----------------+--------+---------+------------+")
    print("| Heuristique    | Coût   | Nœuds   | Temps (s)  |")
    print("+----------------+--------+---------+------------+")
    print(f"| h = 0 (Aucune) | {c_h0:>6} | {n_h0:>7} | {t_h0:>10.4f} |")
    print(f"| Manhattan      | {c_manh:>6} | {n_manh:>7} | {t_manh:>10.4f} |")
    print("+----------------+--------+---------+------------+")

def run_experiment_E4():
    print("\n\n" + "="*60)
    print("=== E.4 : Weighted A* (Grille Difficile) ===")
    print("===       Compromis Vitesse vs Optimalité ===")
    print("="*60)
    
    grilles = create_experiment_grids()
    env = grilles['difficile']
    weights = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    
    print("\n+---------+--------+---------+------------+")
    print("| Poids W | Coût   | Nœuds   | Temps (s)  |")
    print("+---------+--------+---------+------------+")
    
    for w in weights:
        _, c, n, t, _ = graph_search(env, manhattan, 1, w)
        print(f"| {w:>7.1f} | {c:>6} | {n:>7} | {t:>10.4f} |")
        
    print("+---------+--------+---------+------------+\n")

# Nouvelle expérience pour π(n)
def run_experiment_E5():
    print("\n\n" + "="*68)
    print("=== E.5 : Évolution de π(n) pour différents n et ε (Grille Moyenne) ===")
    print("="*68)
    
    grilles = create_experiment_grids()
    env = grilles['moyenne']
    epsilons = [0.0, 0.1, 0.2, 0.3]
    n_steps = [5, 10, 15, 20]
    
    print("\n+------+----------+----------+----------+----------+")
    print("|  ε   | π(5)     | π(10)    | π(15)    | π(20)    |")
    print("+------+----------+----------+----------+----------+")
    
    for eps in epsilons:
        policy = get_policy(env, env.goal)
        P, states, _ = build_transition_matrix(env, policy, eps)
        probs = []
        for n in n_steps:
            pi_n = compute_pi_n(P, states, env.start, env.goal, n)
            probs.append(pi_n)
        print(f"| {eps:>4.1f} | {probs[0]:>8.4f} | {probs[1]:>8.4f} | {probs[2]:>8.4f} | {probs[3]:>8.4f} |")
    
    print("+------+----------+----------+----------+----------+")

# Nouvelle fonction pour analyser classes et périodicité (appelée dans main)
def analyze_markov_classes(eps=0.2):
    print("\n\n" + "="*68)
    print(f"=== Analyse des classes et périodicité (ε = {eps}, Grille Moyenne) ===")
    print("="*68)
    
    grilles = create_experiment_grids()
    env = grilles['moyenne']
    policy = get_policy(env, env.goal)
    P, states, _ = build_transition_matrix(env, policy, eps)
    G = build_transition_graph(P, states)
    classes = identify_classes(G)
    periodicities = check_periodicity(G)
    
    print("\nClasses de communication (composantes fortement connexes):")
    for i, cls in enumerate(classes, 1):
        print(f"Classe {i}: {cls}")
    
    print("\nPériodicités des composantes:")
    for comp, per in periodicities.items():
        print(f"Composante {comp}: Périodicité = {per} (1 = apériodique)")

# === POINT D'ENTRÉE DU SCRIPT ===
if __name__ == "__main__":
    run_experiment_E1()
    run_experiment_E2()
    run_experiment_E3()
    run_experiment_E4()
    run_experiment_E5()  # Nouvelle
    analyze_markov_classes()  # Nouvelle
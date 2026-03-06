import matplotlib.pyplot as plt
import numpy as np
import networkx as nx  # Ajout pour le graphe de transitions

from grid import create_experiment_grids
from astar import graph_search, manhattan, zero_heuristic
from markov import get_policy, build_transition_matrix, simulate_monte_carlo, analyze_absorption, build_transition_graph

def plot_single_grid(ax, env, path, title):
    grille_visuelle = np.zeros((env.height, env.width))
    for (x, y) in env.obstacles:
        grille_visuelle[y, x] = -1 
    
    grille_visuelle[env.start[1], env.start[0]] = 0.8
    grille_visuelle[env.goal[1], env.goal[0]] = 1.0
    
    cmap = plt.cm.get_cmap('terrain')
    ax.imshow(grille_visuelle, cmap=cmap, origin='upper')
    
    ax.set_xticks(np.arange(-.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='both', size=0, labelbottom=False, labelleft=False)
    
    ax.text(env.start[0], env.start[1], 'S', ha='center', va='center', color='black', fontweight='bold')
    ax.text(env.goal[0], env.goal[1], 'G', ha='center', va='center', color='black', fontweight='bold')
    
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color='red', linewidth=3, marker='o', markersize=6, alpha=0.9, label="Chemin A*")
        ax.legend(loc="upper left", fontsize=8)
        
    ax.set_title(title)

def visualiser_toutes_les_grilles():
    grilles = create_experiment_grids()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (nom, env) in zip(axes, grilles.items()):
        chemin_astar, _, _, _, _ = graph_search(env, heuristic_func=manhattan, weight_g=1, weight_h=1)
        plot_single_grid(ax, env, chemin_astar, f"Grille : {nom.capitalize()} ({env.width}x{env.height})")
        
    plt.tight_layout()
    plt.show()

def visualiser_E1_comparaison_noeuds():
    grilles = create_experiment_grids()
    noms = ['facile', 'moyenne', 'difficile']
    
    noeuds_ucs = []
    noeuds_greedy = []
    noeuds_astar = []
    
    for nom in noms:
        env = grilles[nom]
        _, _, n_ucs, _, _ = graph_search(env, heuristic_func=zero_heuristic, weight_g=1, weight_h=0)
        _, _, n_greedy, _, _ = graph_search(env, heuristic_func=manhattan, weight_g=0, weight_h=1)
        _, _, n_astar, _, _ = graph_search(env, heuristic_func=manhattan, weight_g=1, weight_h=1)
        
        noeuds_ucs.append(n_ucs)
        noeuds_greedy.append(n_greedy)
        noeuds_astar.append(n_astar)
    x = np.arange(len(noms))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, noeuds_ucs, width, label='UCS', color='skyblue')
    ax.bar(x, noeuds_greedy, width, label='Greedy', color='lightgreen')
    ax.bar(x + width, noeuds_astar, width, label='A*', color='salmon')
    ax.set_ylabel('Nombre de nœuds explorés')
    ax.set_title('E.1 : Comparaison Nœuds développés')
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in noms])
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualiser_E2_impact_incertitude():
    env = create_experiment_grids()['moyenne']
    
    epsilons = [0.0, 0.1, 0.2, 0.3, 0.4]
    taux_reussite = []
    temps_empiriques = []
    temps_theoriques = []
    
    for eps in epsilons:
        politique = get_policy(env, env.goal)
        P, etats, _ = build_transition_matrix(env, politique, eps)
        
        _, t_theo = analyze_absorption(P, etats, env.start, env.goal)
        temps_theoriques.append(t_theo if t_theo is not None else float('nan'))
        
        reussite, t_emp_succes, _ = simulate_monte_carlo(env, P, etats, env.start, env.goal, N_simulations=1000)
        taux_reussite.append(reussite)
        temps_empiriques.append(t_emp_succes if reussite > 0 else float('nan'))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epsilons, taux_reussite, marker='o', color='purple', linewidth=2)
    ax1.set_xlabel('Niveau d\'incertitude (ε)')
    ax1.set_ylabel('Taux de réussite')
    ax1.set_title('Impact de l\'incertitude sur la réussite')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True)
    
    ax2.plot(epsilons, temps_theoriques, marker='s', linestyle='--', color='blue', label='Théorique')
    ax2.plot(epsilons, temps_empiriques, marker='^', color='orange', label='Empirique')
    ax2.set_xlabel('Niveau d\'incertitude (ε)')
    ax2.set_ylabel('Temps moyen (étapes)')
    ax2.set_title('Temps moyen avant absorption')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualiser_matrice_transition(epsilon=0.2):
    env = create_experiment_grids()['moyenne']
    politique = get_policy(env, env.goal)
    P, etats, _ = build_transition_matrix(env, politique, epsilon)
    
    labels = [str(etat) for etat in etats]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(P, cmap='Blues', vmin=0, vmax=1)
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probabilité de transition')
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    
    ax.set_title(f'Matrice de Transition Stochastique P (ε = {epsilon})')
    ax.set_xlabel('État d\'arrivée (t+1)')
    ax.set_ylabel('État de départ (t)')
    
    plt.tight_layout()
    plt.show()

def visualiser_E4_weighted_astar():
    """
    Génère un graphique montrant le compromis entre Optimalité (Coût) et Vitesse (Nœuds explorés)
    pour Weighted A*.
    """
    print("Génération du graphique E.4 (Weighted A*)...")
    env = create_experiment_grids()['difficile'] # On prend la grille difficile
    weights = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    
    couts = []
    noeuds = []
    
    for w in weights:
        _, c, n, _, _ = graph_search(env, heuristic_func=manhattan, weight_g=1, weight_h=w)
        couts.append(c)
        noeuds.append(n)
        
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    # Axe de gauche : Coût (Optimalité)
    color = 'tab:red'
    ax1.set_xlabel('Poids de l\'heuristique (W)')
    ax1.set_ylabel('Coût du chemin (Optimalité)', color=color)
    ax1.plot(weights, couts, marker='o', color=color, linewidth=2, label="Coût du chemin (Plus bas = Mieux)")
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Axe de droite : Nœuds explorés (Vitesse)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Nœuds explorés (Vitesse)', color=color)  
    ax2.plot(weights, noeuds, marker='s', color=color, linewidth=2, linestyle='--', label="Nœuds explorés (Plus bas = Mieux)")
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  
    plt.title("E.4 : Weighted A* - Compromis Vitesse vs Optimalité")
    fig.legend(loc="center right", bbox_to_anchor=(0.85, 0.5))
    plt.grid(True, alpha=0.3)
    plt.show()

def visualiser_transition_graph(epsilon=0.2):
    print("Génération du graphe des transitions...")
    grilles = create_experiment_grids()
    env = grilles['moyenne']  # Exemple avec grille moyenne
    policy = get_policy(env, env.goal)
    P, states, _ = build_transition_matrix(env, policy, epsilon)
    G = build_transition_graph(P, states)
    
    # Positions simples pour les états (grille 2D)
    pos = {}
    for state in states:
        if isinstance(state, tuple):
            pos[state] = (state[0], state[1])
        else:  # 'FAIL'
            pos[state] = (-1, -1)  # Position arbitraire
    
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=800, font_size=10, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    ax.set_title(f'Graphe Orienté des Transitions (ε = {epsilon})')
    plt.tight_layout()
    plt.show()

# === POINT D'ENTRÉE DU SCRIPT ===
if __name__ == "__main__":
    visualiser_toutes_les_grilles()
    visualiser_E1_comparaison_noeuds()
    visualiser_E2_impact_incertitude()
    visualiser_matrice_transition(epsilon=0.2)
    visualiser_E4_weighted_astar()
    visualiser_transition_graph()  # Nouvelle
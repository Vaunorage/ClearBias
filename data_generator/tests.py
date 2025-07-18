from data_generator.main import generate_data, get_real_data, generate_data_schema, GroupDefinition, \
    generate_from_real_data, generate_optimal_discrimination_data
import matplotlib.pyplot as plt

from data_generator.utils import plot_distribution_comparison, print_distribution_stats, visualize_df, \
    create_parallel_coordinates_plot, plot_and_print_metric_distributions, unique_individuals_ratio, \
    individuals_in_multiple_groups, plot_correlation_matrices, visualize_injected_discrimination
from methods.utils import get_groups

# data, schema = generate_from_real_data('bank')

# %%
# data, schema = get_real_data('bank')

# %%
nb_attributes = 20

schema = generate_data_schema(min_number_of_classes=2, max_number_of_classes=9, prop_protected_attr=0.4,
                              nb_attributes=nb_attributes)

group_defs = [{'alea_uncertainty': 0, 'avg_diff_outcome': 0.43070422535211267, 'diff_subgroup_size': 0.3087557603686636,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 77, 'intersectionality': 3,
               'similarity': 0.71136029765785, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 76, 'Attr3_T': 0, 'Attr4_T': 0}, 'subgroup_bias': 0.43070422535211267},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.15760077707625061, 'diff_subgroup_size': 0.42,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 60, 'intersectionality': 3,
               'similarity': 0.78448433563763, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 54, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.15760077707625061},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.04929577464788732, 'diff_subgroup_size': 0.9722222222222222,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 4, 'intersectionality': 3,
               'similarity': 0.3546967036008132, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 2, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.04929577464788732},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.04329681794470527, 'diff_subgroup_size': 0.136,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 110, 'intersectionality': 3,
               'similarity': 0.8140991269595396, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 28, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.04329681794470527},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.3107042253521127, 'diff_subgroup_size': 0.4791666666666667,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 52, 'intersectionality': 3,
               'similarity': 0.7493530798703212, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 0, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.3107042253521127},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.2102462100849371, 'diff_subgroup_size': 0.297029702970297,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 264, 'intersectionality': 3,
               'similarity': 0.8483279698259456, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 8, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.2102462100849371},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.3624689312344656, 'diff_subgroup_size': 0.6136363636363636,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 36, 'intersectionality': 3,
               'similarity': 0.6314981210502145, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 76, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.3624689312344656},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.028290432248664406,
               'diff_subgroup_size': 0.10077519379844961, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 118, 'intersectionality': 3, 'similarity': 0.8132114461509702,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 34, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.028290432248664406},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.12051554610682966, 'diff_subgroup_size': 0.4564102564102564,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 55, 'intersectionality': 3,
               'similarity': 0.7517154126534991, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 48, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.12051554610682966},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.11737089201877934, 'diff_subgroup_size': 0.918918918918919,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 8, 'intersectionality': 3,
               'similarity': 0.3763163863163863, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 8, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.11737089201877934},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.016124786099776224,
               'diff_subgroup_size': 0.14056224899598393, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 109, 'intersectionality': 3, 'similarity': 0.8435842339345314,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 42, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.016124786099776224},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.03403755868544601, 'diff_subgroup_size': 0.8441558441558441,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 14, 'intersectionality': 3,
               'similarity': 0.5588517465440542, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 69, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.03403755868544601},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.12129246064623034,
               'diff_subgroup_size': 0.08974358974358974, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 172, 'intersectionality': 3, 'similarity': 0.8514200704941846,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 34, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.12129246064623034},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.030526841760094937, 'diff_subgroup_size': 0.521079258010118,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 453, 'intersectionality': 3,
               'similarity': 0.8641161375156412, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 44, 'Attr3_T': 1, 'Attr4_T': 1}, 'subgroup_bias': 0.030526841760094937},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.07836379982019778, 'diff_subgroup_size': 0.2033898305084746,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 96, 'intersectionality': 3,
               'similarity': 0.8319162988846247, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 52, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.07836379982019778},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.09023910907304292, 'diff_subgroup_size': 0.5351351351351351,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 45, 'intersectionality': 3,
               'similarity': 0.6847504648317655, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 34, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.09023910907304292},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.029135597901132286,
               'diff_subgroup_size': 0.47150259067357514, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 53, 'intersectionality': 3, 'similarity': 0.6621372123811149,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 37, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.029135597901132286},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.05676483141271874,
               'diff_subgroup_size': 0.36538461538461536, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 68, 'intersectionality': 3, 'similarity': 0.7896464646464647,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 46, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.05676483141271874},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.022132796780684104,
               'diff_subgroup_size': 0.8205128205128205, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 16, 'intersectionality': 3, 'similarity': 0.5437233337233338,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 11, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.022132796780684104},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.17797695262483995, 'diff_subgroup_size': 0.7317073170731707,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 24, 'intersectionality': 3,
               'similarity': 0.5882827937932567, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 73, 'Attr3_T': 2, 'Attr4_T': 1}, 'subgroup_bias': 0.17797695262483995},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.16737089201877936,
               'diff_subgroup_size': 0.08396946564885496, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 122, 'intersectionality': 3, 'similarity': 0.7506812668616382,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 2, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.16737089201877936},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.06677565392354126,
               'diff_subgroup_size': 0.11811023622047244, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 114, 'intersectionality': 3, 'similarity': 0.7552870152270003,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 46, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.06677565392354126},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.23641851106639838, 'diff_subgroup_size': 0.5434782608695652,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 44, 'intersectionality': 3,
               'similarity': 0.787955968657723, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 56, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.23641851106639838},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.03267143846686677,
               'diff_subgroup_size': 0.39901477832512317, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 63, 'intersectionality': 3, 'similarity': 0.7334588488434642,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 54, 'Attr3_T': 2, 'Attr4_T': 1}, 'subgroup_bias': 0.03267143846686677},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.09586551567469334,
               'diff_subgroup_size': 0.39215686274509803, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 64, 'intersectionality': 3, 'similarity': 0.7553768453768454,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 73, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.09586551567469334},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.01679577464788732,
               'diff_subgroup_size': 0.47601476014760147, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 402, 'intersectionality': 3, 'similarity': 0.8740561546845651,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 50, 'Attr3_T': 1, 'Attr4_T': 1}, 'subgroup_bias': 0.01679577464788732},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.10093896713615025, 'diff_subgroup_size': 0.2,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 215, 'intersectionality': 3,
               'similarity': 0.8549744077312007, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 31, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.10093896713615025},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.050704225352112685,
               'diff_subgroup_size': 0.14457831325301204, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 192, 'intersectionality': 3, 'similarity': 0.8590496849175461,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 37, 'Attr3_T': 2, 'Attr4_T': 1}, 'subgroup_bias': 0.050704225352112685},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.059795134443021765,
               'diff_subgroup_size': 0.12698412698412698, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 112, 'intersectionality': 3, 'similarity': 0.8376791080405539,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 50, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.059795134443021765},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.15825139516343345, 'diff_subgroup_size': 0.4564102564102564,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 55, 'intersectionality': 3,
               'similarity': 0.7618903001754652, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 28, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.15825139516343345},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.061815336463223784,
               'diff_subgroup_size': 0.4489795918367347, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 56, 'intersectionality': 3, 'similarity': 0.8057611379764033,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 48, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.061815336463223784},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.06498993963782695, 'diff_subgroup_size': 0.6045197740112994,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 37, 'intersectionality': 3,
               'similarity': 0.6829799474030244, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 42, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.06498993963782695},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.0033358042994810974,
               'diff_subgroup_size': 0.19831223628691982, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 97, 'intersectionality': 3, 'similarity': 0.8275829715901272,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 56, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.0033358042994810974},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.09615877080665813, 'diff_subgroup_size': 0.4416243654822335,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 57, 'intersectionality': 3,
               'similarity': 0.7173931647642627, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 50, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.09615877080665813},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.12738267058179467,
               'diff_subgroup_size': 0.33176470588235296, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 285, 'intersectionality': 3, 'similarity': 0.8464805256502669,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 28, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.12738267058179467},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.07943985753602073, 'diff_subgroup_size': 0.5077989601386482,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 437, 'intersectionality': 3,
               'similarity': 0.8765406118847415, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 56, 'Attr3_T': 1, 'Attr4_T': 1}, 'subgroup_bias': 0.07943985753602073},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.07070422535211268, 'diff_subgroup_size': 0.0273972602739726,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 152, 'intersectionality': 3,
               'similarity': 0.821877681606741, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 37, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.07070422535211268},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.1825224071702945, 'diff_subgroup_size': 0.2154696132596685,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 222, 'intersectionality': 3,
               'similarity': 0.8457147798336895, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 5, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.1825224071702945},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.08795912731289701,
               'diff_subgroup_size': 0.16393442622950818, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 104, 'intersectionality': 3, 'similarity': 0.8089477766608587,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 37, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.08795912731289701},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.12091699130955949, 'diff_subgroup_size': 0.5026455026455027,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 49, 'intersectionality': 3,
               'similarity': 0.7523120915866381, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 50, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.12091699130955949},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.03548683404776485, 'diff_subgroup_size': 0.5282392026578073,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 462, 'intersectionality': 3,
               'similarity': 0.8760588769116914, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 52, 'Attr3_T': 1, 'Attr4_T': 1}, 'subgroup_bias': 0.03548683404776485},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.010704225352112677,
               'diff_subgroup_size': 0.4791666666666667, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 52, 'intersectionality': 3, 'similarity': 0.7238567700106162,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 46, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.010704225352112677},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.2234314980793854, 'diff_subgroup_size': 0.4416243654822335,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 57, 'intersectionality': 3,
               'similarity': 0.8050263963678598, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 74, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.2234314980793854},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.13019140483929217, 'diff_subgroup_size': 0.4239350912778905,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 353, 'intersectionality': 3,
               'similarity': 0.8438019669704231, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 22, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.13019140483929217},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.004694835680751179, 'diff_subgroup_size': 0.5,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 428, 'intersectionality': 3,
               'similarity': 0.8717635664873487, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 46, 'Attr3_T': 1, 'Attr4_T': 1}, 'subgroup_bias': 0.004694835680751179},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.09964039556487864, 'diff_subgroup_size': 0.5026455026455027,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 49, 'intersectionality': 3,
               'similarity': 0.6967602597395743, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 54, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.09964039556487864},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.18241154242528343,
               'diff_subgroup_size': 0.26785714285714285, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 84, 'intersectionality': 3, 'similarity': 0.858112678961647,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 42, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.18241154242528343},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.02362089201877935,
               'diff_subgroup_size': 0.19327731092436976, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 98, 'intersectionality': 3, 'similarity': 0.8124793155562388,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 25, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.02362089201877935},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.007629107981220656,
               'diff_subgroup_size': 0.32710280373831774, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 74, 'intersectionality': 3, 'similarity': 0.8078496503496505,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 52, 'Attr3_T': 2, 'Attr4_T': 1}, 'subgroup_bias': 0.007629107981220656},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.19971608305962257, 'diff_subgroup_size': 0.2810126582278481,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 255, 'intersectionality': 3,
               'similarity': 0.8414470145239377, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 6, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.19971608305962257},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.04685807150595884,
               'diff_subgroup_size': 0.15447154471544716, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 106, 'intersectionality': 3, 'similarity': 0.8109497340266572,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 54, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.04685807150595884},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.4507042253521127, 'diff_subgroup_size': 0.6705882352941176,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 30, 'intersectionality': 3,
               'similarity': 0.7102637433541618, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 75, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.4507042253521127},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.050704225352112685,
               'diff_subgroup_size': 0.4791666666666667, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 52, 'intersectionality': 3, 'similarity': 0.7179248026667694,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 39, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.050704225352112685},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.09324320085322627, 'diff_subgroup_size': 0.5194585448392555,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 451, 'intersectionality': 3,
               'similarity': 0.8614532020816127, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 15, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.09324320085322627},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.05487089201877935,
               'diff_subgroup_size': 0.49473684210526314, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 50, 'intersectionality': 3, 'similarity': 0.7261618428727181,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 44, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.05487089201877935},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.20924081071796635,
               'diff_subgroup_size': 0.18155619596541786, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 207, 'intersectionality': 3, 'similarity': 0.8378333932274434,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 3, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.20924081071796635},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.07026944274341704,
               'diff_subgroup_size': 0.21367521367521367, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 94, 'intersectionality': 3, 'similarity': 0.8063927319719719,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 48, 'Attr3_T': 2, 'Attr4_T': 1}, 'subgroup_bias': 0.07026944274341704},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.07861120209629874,
               'diff_subgroup_size': 0.24561403508771928, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 88, 'intersectionality': 3, 'similarity': 0.830091484126175,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 50, 'Attr3_T': 2, 'Attr4_T': 1}, 'subgroup_bias': 0.07861120209629874},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.12756012054861923, 'diff_subgroup_size': 0.5266666666666666,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 460, 'intersectionality': 3,
               'similarity': 0.8624112342317906, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 11, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.12756012054861923},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.09704568876674682,
               'diff_subgroup_size': 0.07169811320754717, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 125, 'intersectionality': 3, 'similarity': 0.8178514208688594,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 39, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.09704568876674682},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.04929577464788732, 'diff_subgroup_size': 0.832258064516129,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 15, 'intersectionality': 3,
               'similarity': 0.5754778554778555, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 74, 'Attr3_T': 2, 'Attr4_T': 1}, 'subgroup_bias': 0.04929577464788732},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.04929577464788732, 'diff_subgroup_size': 0.8933333333333333,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 10, 'intersectionality': 3,
               'similarity': 0.43789654789654786, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 73, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.04929577464788732},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.007522407170294496,
               'diff_subgroup_size': 0.23478260869565218, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 90, 'intersectionality': 3, 'similarity': 0.7832944832944833,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 48, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.007522407170294496},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.1022193768672642, 'diff_subgroup_size': 0.36538461538461536,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 68, 'intersectionality': 3,
               'similarity': 0.7598082428738443, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 31, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.1022193768672642},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.018886043533930856,
               'diff_subgroup_size': 0.5268817204301075, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 46, 'intersectionality': 3, 'similarity': 0.7348974893127974,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 25, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.018886043533930856},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.0770200148257969, 'diff_subgroup_size': 0.33489461358313816,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 287, 'intersectionality': 3,
               'similarity': 0.8498119559253163, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 25, 'Attr3_T': 2, 'Attr4_T': 2}, 'subgroup_bias': 0.0770200148257969},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.02477829942618675, 'diff_subgroup_size': 0.4489795918367347,
               'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0, 'group_size': 56, 'intersectionality': 3,
               'similarity': 0.682560432803431, 'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 57, 'Attr3_T': 0, 'Attr4_T': 2}, 'subgroup_bias': 0.02477829942618675},
              {'alea_uncertainty': 0, 'avg_diff_outcome': 0.039199800573351616,
               'diff_subgroup_size': 0.11372549019607843, 'epis_uncertainty': 0, 'frequency': 1, 'granularity': 0,
               'group_size': 115, 'intersectionality': 3, 'similarity': 0.8072212378000364,
               'subgroup1': {'Attr1_T': 68, 'Attr3_T': 1, 'Attr4_T': 0},
               'subgroup2': {'Attr1_T': 39, 'Attr3_T': 0, 'Attr4_T': 1}, 'subgroup_bias': 0.039199800573351616}]
predefined_groups = [GroupDefinition(**e) for e in group_defs]

# data_obj_synth, schema = generate_from_real_data('bank', nb_groups=10,
#                                                  predefined_groups=predefined_groups[:4],
#                                                  use_cache=False)

# predefined_groups_origin, nb_elements = get_groups(data_obj_synth, data, schema)

# %%
data1 = generate_optimal_discrimination_data(
    nb_groups=100,
    nb_attributes=15,
    prop_protected_attr=0.3
)

# data1 = generate_data(
#     nb_attributes=5,
#     min_number_of_classes=3,
#     max_number_of_classes=5,
#     nb_groups=100,
#     max_group_size=100,
#     categorical_outcome=True,
#     nb_categories_outcome=4,
#     corr_matrix_randomness=1,
#     categorical_influence=1,
#     # data_schema=schema,
#     use_cache=False,
#     min_similarity=0.9,
#     max_similarity=1.0
#     # predefined_groups=predefined_groups,
#     # additional_random_rows=30000
# )

print(f"Generated {len(data1.dataframe)} samples in {data1.nb_groups} groups")

#%%
data_obj = generate_optimal_discrimination_data(
    nb_groups=50,
    nb_attributes=10,
    prop_protected_attr=0.3,
    min_group_size=50,
    max_group_size=200,
    categorical_outcome=True # Continuous outcomes are easier to visualize
)

# 2. Visualize the results
print("\nCreating visualizations...")
visualize_injected_discrimination(
    data=data_obj,
    sample_size=10000,
    top_n_biased_groups=4
)
# %%
import pandas as pd
from path import HERE

df = pd.concat([data1.xdf, data1.ydf], axis=1)
fig = visualize_df(df, data1.attr_columns, data1.outcome_column, HERE.joinpath('ll.png'))
fig.show()

# %%
create_parallel_coordinates_plot(data1.dataframe)
plt.show()

# Print statistics
print_distribution_stats(schema, data1)

# %%

plot_and_print_metric_distributions(data1.dataframe)

# %%
# Example usage:
individual_col = 'indv_key'
group_col = 'group_key'

unique_ratio, duplicates_count, total = unique_individuals_ratio(data1.dataframe, 'indv_key', data1.attr_possible_values)
individuals_in_multiple_groups_count = individuals_in_multiple_groups(data1.dataframe, individual_col, group_col)

print(f"Unique Individuals Ratio: {unique_ratio}, duplicate : {duplicates_count}, total: {total}")

# %%
plot_correlation_matrices(schema.correlation_matrix, data1)



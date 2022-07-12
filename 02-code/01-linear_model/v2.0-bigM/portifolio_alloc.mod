# *****************************************************************************
# Portifolio Allocation Problem												  *
# --------------------------------------------------------------------------- *
# The ideia here is sugest a portifolio of Investiment Funds Assets           *
# negotiated B3 (Brazilian Stock Exchange). This is a optimization problem    *
# where we try to optimize the allocation seeking performance (represented by *
# dividend yield) subject mainly to the given budget.                         *
# Other features of this modeling may include other financial objectives such *
# as diversity, risk minimization and so on.                                  *
# --------------------------------------------------------------------------- *
# Lucas Lukasavicus                                                           *
# 09 apr 2022                                                                 *
# lukasavicus at gmail dot com                                                * 
# *****************************************************************************

# === PARAMETERS ==============================================================
# indexes
param n_assets; 							# Number of Assets;

# sets
set AS := {1..n_assets};					# Assets set;

# arrays
param asset_price{AS}; 		 				# ...
param asset_dy{AS}; 						# ...
param asset_lqdt{AS}; 						# ... 
param asset_pvp{AS}; 						# ...  
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === DECISION VARIABLES ======================================================
var qa{AS} integer >= 0;					# quantity of ...;
var qab{AS} binary;							# If that is some investiment on that asset ...;
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === USER PARAMETERS =========================================================
param BUDGET = 1000.00;

param LIQUIDITY_IMPORTANCE = 10;
param DIVERSITY_IMPORTANCE = 1;

param MIN_PVP = 0.9;
param MAX_PVP = 1.4;
param MAX_CONCENTRATION_PERCENTAGE = 0.50;
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === HELPER VARIABLES ========================================================
param BIG_M = BUDGET * BUDGET;
var profit = sum {i in AS} (qa[i] * asset_dy[i]);
var expend_per_asset{i in AS} =  qa[i] * asset_price[i];
var expend = sum {i in AS} qa[i] * asset_price[i];

var rstr_lqdt{i in AS} = (if qa[i] > 0 then 1 else 0) * asset_lqdt[i];

#var diversity = sum {i in AS} ((if qa[i] > 0 then 1 else 0) );
#var qab{i in AS} = if qa[i] > 0 then 1 else 0;

# --- VAR: Métricas ----------------------------------------------------------
# Sejam as carteiras:
# Seja a carteira 1: (A: 100, B: 20, C: 0)
# Seja a carteira 2: (A: 70, B: 20, C: 10)
# ---
# L1: Liquidez deveria ser uma medida que diz: quanto maior for o número de
# assets maior a liquidez, por exemplo 100 A é uma carteira mais diversa do
# que 50 B com o mesmo budget.
# ---
# D1: Uma forma de ver a diversidade seria ver quantos assets diferentes
# compõem a carteira.
# ---
# X | L1  | D1
# c1| 120 | 2
# c2| 100 | 3
# ---


var liquidity = sum {i in AS} qa[i];
var diversity = sum {i in AS} qab[i];

# --- atempts: ----------------------------------------------------------------
#var rstr_lqdt{i in AS} = 
#	( ((if qa[i] > 0 then 1 else 0) * asset_lqdt[i]) >= 1 );
#var LB = sum {i in HS} qh[i] * rh[i]; 		# Lucro Bruto
#var C =  sum {i in HS} qh[i] * ah[i] * 10;	# Costs
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# === OBJECTIVE FUNCTION ======================================================

maximize Z:
	profit +
	+ (liquidity * LIQUIDITY_IMPORTANCE)
	+ (diversity * DIVERSITY_IMPORTANCE); 

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === RESTRICTIONS ============================================================
s.t. Rstr_Budget: sum {i in AS} qa[i] * asset_price[i] <= BUDGET;

s.t. internal_restriction_1 {i in AS}: qab[i] <= qa[i];
s.t. internal_restriction_2 {i in AS}: qa[i] <= BIG_M*qab[i];

s.t. Rstr_lqdt {i in AS}: asset_lqdt[i] >= qab[i];

s.t. Rstr_min_pvp {i in AS}: qab[i] * MIN_PVP <= asset_pvp[i];
s.t. Rstr_max_pvp {i in AS}: qab[i] * MAX_PVP >= qab[i] *asset_pvp[i];

s.t. Rstr_concentration {i in AS}: qa[i] * asset_price[i] <= BUDGET * MAX_CONCENTRATION_PERCENTAGE;

#s.t. Rstr_2 {i in AS}: qa[i] * asset_lqdt[i] > 0;


# --- examples: ---------------------------------------------------------------
#s.t. Rstr_3: sum {i in AS} qa[i] * asset_price[i] <= BUDGET;
#s.t. Rstr_3: qh[2] >= ((1/2) * (qh[1]+qh[3]));
#s.t. Rstr_4: sum {i in HS} qh[i] * ah[i] <= 10000;
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# *****************************************************************************
# OFFICIAL DISCLAIMER                                                         *
# --------------------------------------------------------------------------- *
# Os instrumentos financeiros negociados em Bolsa de Valores podem não ser    *
# adequados para todos os investidores. Os resultados disponibilizados por    *
# esse programa não levam em consideração os objetivos de investimento, a     *
# situação financeira ou as necessidades específicas de um determinado        *
# investidor. A decisão final em relação aos investimentos deve ser tomada    *
# por cada investidor, levando em consideração os vários riscos, tarifas e    *
# comissões. A rentabilidade de instrumentos financeiros pode apresentar      *
# variações, e seu preço ou valor pode aumentar ou diminuir, podendo,         *
# dependendo das características da operação, implicar em perdas de valor até *
# superior ao capital investido.                                              *
# Os desempenhos anteriores não são indicativos de resultados futuros e       *
# nenhuma declaração ou garantia, de forma expressa ou implícita, é feita em  *
# relação a desempenhos futuros. O investimento em renda variável é           *
# considerado de alto risco, podendo ocasionar perdas, inclusive, superiores  *
# ao montante de capital alocado. O prejuízo potencial apresentado é          *
# meramente indicativo e o prejuízo efetivamente decorrente das operações     *
# realizadas pode ser substancialmente distinto da estimativa. Os preços e    *
# disponibilidades dos instrumentos financeiros são meramente indicativos e   *
# sujeitos a alterações sem aviso prévio. O investidor que realiza operações  *
# de renda variável é o único responsável pelas decisões de investimento ou   *
# de abstenção de investimento que tomar.                                     *
# *****************************************************************************
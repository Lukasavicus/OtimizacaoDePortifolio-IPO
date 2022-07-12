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
#set NP := {'price', 'dy', 'lqdt', 'pvp'};	# Paramenters;
set NP := {'price','lqdt','divd','dy','dy3ac','dy6ac','dy12ac','dy3m','dy6m','dy12m','dyy','price_delta','profitability','profitability_ac','liq_patr','vpa','pvp','dyp','patr_delta','sector_none','sector_hosp','sector_hotel','sector_hbd','sector_corp','sector_log','sector_other','sector_resd','sector_shop','sector_title'};
set sectors := {'sector_none','sector_hosp','sector_hotel','sector_hbd','sector_corp','sector_log','sector_other','sector_resd','sector_shop','sector_title'};

# arrays
param asset_data{AS, NP}; 		 				# ...
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === DECISION VARIABLES ======================================================
var qa{AS} integer >= 0;					# quantity of ...;
var qab{AS} binary;							# If that is some investiment on that asset ...;
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === USER PARAMETERS =========================================================
param BUDGET = 10000.00;

param LIQUIDITY_IMPORTANCE = 10;
param DIVERSITY_IMPORTANCE = 10;

param MIN_PVP = 0.6;
param MAX_PVP = 1.4;
param MAX_CONCENTRATION_PERCENTAGE = 0.30;
param MAX_CONCENTRATION_PERCENTAGE_INTER_SECTOR = 0.30;
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === HELPER VARIABLES ========================================================
#param BIG_M = BUDGET * BUDGET;
param BIG_M = 1000000000000;
var profit = sum {i in AS} (qa[i] * asset_data[i, 'dy']);
var expend_per_asset{i in AS} =  qa[i] * asset_data[i, 'price'];
var expend = sum {i in AS} qa[i] * asset_data[i, 'price'];

var rstr_lqdt{i in AS} = (if qa[i] > 0 then 1 else 0) * asset_data[i, 'lqdt'];

#var rstr_diversity_inter_domains;
var n_assets_per_sector {s in sectors} = sum {i in AS} (qa[i] * asset_data[i, s]);
var n_assets_per_sector_value {s in sectors} = sum {i in AS} (qa[i] * asset_data[i, 'price'] * asset_data[i, s]);
var n_assets_per_sector_bin {s in sectors} = sum {i in AS} (qab[i] * asset_data[i, s]);

#var diversity = sum {i in AS} ((if qa[i] > 0 then 1 else 0) );
#var qab{i in AS} = if qa[i] > 0 then 1 else 0;

# --- VAR: M�tricas ----------------------------------------------------------
# Sejam as carteiras:
# Seja a carteira 1: (A: 100, B: 20, C: 0)
# Seja a carteira 2: (A: 70, B: 20, C: 10)
# ---
# L1: Liquidez deveria ser uma medida que diz: quanto maior for o n�mero de
# assets maior a liquidez, por exemplo 100 A � uma carteira mais diversa do
# que 50 B com o mesmo budget.
# ---
# D1: Uma forma de ver a diversidade seria ver quantos assets diferentes
# comp�em a carteira.
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
	#- risk

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === RESTRICTIONS ============================================================
s.t. Rstr_Budget: sum {i in AS} qa[i] * asset_data[i, 'price'] <= BUDGET;

s.t. internal_restriction_1 {i in AS}: qab[i] <= qa[i];
s.t. internal_restriction_2 {i in AS}: qa[i] <= BIG_M*qab[i];

s.t. Rstr_lqdt {i in AS}: asset_data[i, 'lqdt'] >= qab[i];

s.t. Rstr_min_pvp {i in AS}: qab[i] * MIN_PVP <= asset_data[i, 'pvp'];
s.t. Rstr_max_pvp {i in AS}: qab[i] * MAX_PVP >= qab[i] *asset_data[i, 'pvp'];

s.t. Rstr_concentration {i in AS}: qa[i] * asset_data[i, 'price'] <= BUDGET * MAX_CONCENTRATION_PERCENTAGE;

s.t. Rstr_concentration_inter_sector {s in sectors}: n_assets_per_sector_value[s] <= BUDGET * MAX_CONCENTRATION_PERCENTAGE_INTER_SECTOR;

#s.t. Rstr_2 {i in AS}: qa[i] * asset_lqdt[i] > 0;


# --- examples: ---------------------------------------------------------------
#s.t. Rstr_3: sum {i in AS} qa[i] * asset_price[i] <= BUDGET;
#s.t. Rstr_3: qh[2] >= ((1/2) * (qh[1]+qh[3]));
#s.t. Rstr_4: sum {i in HS} qh[i] * ah[i] <= 10000;
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# *****************************************************************************
# OFFICIAL DISCLAIMER                                                         *
# --------------------------------------------------------------------------- *
# Os instrumentos financeiros negociados em Bolsa de Valores podem n�o ser    *
# adequados para todos os investidores. Os resultados disponibilizados por    *
# esse programa n�o levam em considera��o os objetivos de investimento, a     *
# situa��o financeira ou as necessidades espec�ficas de um determinado        *
# investidor. A decis�o final em rela��o aos investimentos deve ser tomada    *
# por cada investidor, levando em considera��o os v�rios riscos, tarifas e    *
# comiss�es. A rentabilidade de instrumentos financeiros pode apresentar      *
# varia��es, e seu pre�o ou valor pode aumentar ou diminuir, podendo,         *
# dependendo das caracter�sticas da opera��o, implicar em perdas de valor at� *
# superior ao capital investido.                                              *
# Os desempenhos anteriores n�o s�o indicativos de resultados futuros e       *
# nenhuma declara��o ou garantia, de forma expressa ou impl�cita, � feita em  *
# rela��o a desempenhos futuros. O investimento em renda vari�vel �           *
# considerado de alto risco, podendo ocasionar perdas, inclusive, superiores  *
# ao montante de capital alocado. O preju�zo potencial apresentado �          *
# meramente indicativo e o preju�zo efetivamente decorrente das opera��es     *
# realizadas pode ser substancialmente distinto da estimativa. Os pre�os e    *
# disponibilidades dos instrumentos financeiros s�o meramente indicativos e   *
# sujeitos a altera��es sem aviso pr�vio. O investidor que realiza opera��es  *
# de renda vari�vel � o �nico respons�vel pelas decis�es de investimento ou   *
# de absten��o de investimento que tomar.                                     *
# *****************************************************************************
# EDeMOS (Electricity Demand Mapping from Open-Source data) __KENYA__ case study
<pre>
</pre>
## Crete a new environment in Conda

The attached _*.yml_ will allow you to create a new environment in Conda. The new environment has been tested and is working (there are no issues with libraries like gdal, h3, h3pandas, and others).  
```
conda env create -f environment.yml
```

## Useful data sets

1. Transmission lines 220 kV, 132 kV, others.
[Kenya electricity network](https://energydata.info/dataset/kenya-kenya-electricity-network)
2. Kenya Population and Housing Census 2019. [KNBS, 2019](https://open.africa/dataset/2019-kenya-population-and-housing-census).
3. Demographic and Health Surveys (DHS). [DHS](https://dhsprogram.com/data/dataset/Kenya_Standard-DHS_2022.cfm)
4. A high-resolution gridded dataset to assess electrification in sub-Saharan Africa [^1].
5. Gridded global Gross Domestic Product and Human Development Index datasets over 1990–2015 [^2]. 
6. High-Resolution Electricity Access. set_lightscore_sy_xxxx.tif: Predicted likelihood that a settlement is electrified (0 to 1) [^3].
7. Relative Wealth Index (RWI) [^4].
8. Gridded population estimates from building footprints [^5].

## Cite this work

The work corcerning for EDeMOs Kenya has been carried out by [Roberto Heredia-Fonseca](https://profiles.imperial.ac.uk/r.heredia-fonseca). This work is based on the work EDeMOS Zambia [^6].

[^1]: Falchetta, G., Pachauri, S., Parkinson, S. et al. A high-resolution gridded dataset to assess electrification in sub-Saharan Africa. Sci Data 6, 110 (2019). https://doi.org/10.1038/s41597-019-0122-6.
[^2]: Kummu, M., Taka, M. & Guillaume, J. Gridded global datasets for Gross Domestic Product and Human Development Index over 1990–2015. Sci Data 5, 180004 (2018). https://doi.org/10.1038/sdata.2018.4.
[^3]: Brian Min, Zachary P. O'Keeffe, Babatunde Abidoye, Kwawu Mensan Gaba, Trevor Monroe, Benjamin P. Stewart, Kim Baugh, Bruno Sánchez-Andrade Nuño, “Lost in the Dark: A Survey of Energy Poverty from Space,” Joule (2024), https://doi.org/10.1016/j.joule.2024.05.001.
[^4]: Samapriya Roy, Swetnam, T., & Saah, A. (2025). samapriya/awesome-gee-community-datasets: Community Catalog (3.2.0).
Zenodo. https://doi.org/10.5281/zenodo.14757583.
[^5]: Leasure DR, Dooley CA, Bondarenko M, Tatem AJ. 2021. peanutButter: An R package to produce rapid-response gridded population estimates from building footprints, version 1.0.0. WorldPop, University of Southampton. doi: 10.5258/SOTON/WP00717. https://github.com/wpgp/peanutButter
[^6]: Millot, A., Kerekeš, A., Korkovelos, A., Stringer M., and Hawkes A. EDeMOS_Zambia. GitHub repository. Accessed February 10, 2025. https://github.com/ariane-millot/EDeMOS_Zambia.
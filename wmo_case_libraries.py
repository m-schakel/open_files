def get_cbs_metadata( dataset ):
  # Purpose: Getting and printing CBS metadata for a specific datasets (e.g. 84751NED)
  #          Info and available datasets: https://opendata.cbs.nl/statline/portal.html?_la=nl
  # Example of execution: check_columns( { 2019: '84753NED', 2020: '84908NED' } )
  # Arguments:
  #   dataset              = stringvalue that refers to the CBS dataset (e.g. '84908NED')
  # Returns:
  #   l_metadata           = dataframe with metadata of the given dataset. Some details are printed as well.

  # Downloading table list
  #toc = pd.DataFrame(cbsodata.get_table_list())

  # Downloading metadata
  l_metadata = pd.DataFrame(cbsodata.get_meta(dataset, 'DataProperties'))
  print(l_metadata[['Key','Title','Type']])


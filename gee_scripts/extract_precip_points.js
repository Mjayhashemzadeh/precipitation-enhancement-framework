// 1. Define your stations as FeatureCollection
var stations = ee.FeatureCollection("your_station_points");

// Generate monthly date list
var makeMonthlyList = function(start, end) {
  var nMonths = ee.Date(end).difference(ee.Date(start), 'month').subtract(1);
  var months = ee.List.sequence(0, nMonths);
  return months.map(function(m) {
    var startDate = ee.Date(start).advance(m, 'month');
    var endDate = startDate.advance(1, 'month');
    return ee.Dictionary({
      start: startDate.format('YYYY-MM-dd'),
      end: endDate.format('YYYY-MM-dd'),
      monthStr: startDate.format('YYYY-MM')
    });
  });
};

var dateList = makeMonthlyList('2014-01-01', '2025-01-01');

// Function to resample image to 1km resolution
var resampleTo1km = function(image) {
  return image.resample('bilinear').reproject({
    crs: image.projection(),
    scale: 1000
  });
};

// General function to extract monthly precipitation with 1km resampling
var extractMonthlyPrecip = function(product, reducer, unitConversion) {
  return ee.FeatureCollection(dateList.iterate(function(d, prev) {
    d = ee.Dictionary(d);
    var start = ee.Date(d.get('start'));
    var end = ee.Date(d.get('end'));
    var monthStr = d.get('monthStr');

    var image = product.filterDate(start, end).reduce(reducer)
      .multiply(unitConversion)
      .rename('precip_mm');

    // Resample to 1km before sampling
    var resampled = resampleTo1km(image);

    var sampled = resampled.sampleRegions({
      collection: stations,
      scale: 1000,
      projection: resampled.projection(),
      geometries: true
    }).map(function(f) {
      return f.set('month', monthStr);
    });

    return ee.FeatureCollection(prev).merge(sampled);
  }, ee.FeatureCollection([])));
};

// === TerraClimate ===
var terra = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select("pr"); // mm/month
var terraPrecip = extractMonthlyPrecip(terra, ee.Reducer.mean(), 1);

// === ERA5-Land (DAILY_AGGR, daily total precipitation in meters → mm/month) ===
var era5_daily = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select("total_precipitation_sum");

var era5Precip = ee.FeatureCollection(dateList.iterate(function(d, prev) {
  d = ee.Dictionary(d);
  var start = ee.Date(d.get('start'));
  var end = ee.Date(d.get('end'));
  var monthStr = d.get('monthStr');

  // Sum daily precipitation over the month (units: meters)
  var image = era5_daily.filterDate(start, end)
                       .sum()
                       .multiply(1000)  // Convert meters to millimeters
                       .rename('precip_mm');

  // Resample to 1km before sampling
  var resampled = resampleTo1km(image);

  var sampled = resampled.sampleRegions({
    collection: stations,
    scale: 1000,
    projection: resampled.projection(),
    geometries: true
  }).map(function(f) {
    return f.set('month', monthStr);
  });

  return ee.FeatureCollection(prev).merge(sampled);
}, ee.FeatureCollection([])));

// === EXPORT to CSV ===
Export.table.toDrive({
  collection: terraPrecip,
  description: 'TerraClimate_Precip_2014_2024_1km',
  folder: 'GEE_Exports',
  fileNamePrefix: 'terra_precip_monthly_1km',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: era5Precip,
  description: 'ERA5Land_Precip_2014_2024_1km',
  folder: 'GEE_Exports',
  fileNamePrefix: 'era5_precip_monthly_1km',
  fileFormat: 'CSV'
});

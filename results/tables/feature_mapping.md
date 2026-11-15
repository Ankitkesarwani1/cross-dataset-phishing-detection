# Feature Alignment Mapping Table

| concept | phi_col | uci_col | transform | available | confidence |
| --- | --- | --- | --- | --- | --- |
| IP_in_URL | IsDomainIP | having_IP_Address | PhiUSIIL: 1→−1 (phishy), 0→1 (legit). Direct binary match. | URL-only | high |
| URL_Length | URLLength | URL_Length | PhiUSIIL is numeric char count. Recode: ≥75 →−1, 54–74 →0, <54 →1. | URL-only | high |
| Sub_Domain | NoOfSubDomain | having_Sub_Domain | Recode dot-count: 1 dot →1 (legit), 2 dots →0, ≥3 →−1. | URL-only | high |
| SSL_State | IsHTTPS | SSLfinal_State | PhiUSIIL: IsHTTPS=1 (has HTTPS) →1; 0 →−1. UCI also uses cert issuer and age (0 = suspicious); approximate. | URL-only (scheme) / page-derived (cert details) | medium |
| Favicon | HasFavicon | Favicon | PhiUSIIL: 1 (has favicon from same domain) →1; 0 →−1. | page-derived | high |
| Iframe | NoOfiFrame | Iframe | PhiUSIIL: count > 0 →−1 (has iframe, suspicious); 0 →1. | page-derived | high |
| Popup | NoOfPopup | popUpWidnow | PhiUSIIL: count > 0 →−1 (popup present); 0 →1. | page-derived | high |
| Redirect | NoOfURLRedirect | Redirect | PhiUSIIL max is 1, so using count ≥ 1 → 1; 0 → 0. UCI Redirect is 0/1 (≤1 redirect vs ≥2). PhiUSIIL only records 0 or 1 redirects so this is approximate. | page-derived | low |
| Has_External_FormSubmit | HasExternalFormSubmit | SFH | PhiUSIIL: 1 (form posts to external domain) →−1; 0 →1. UCI SFH also has 0 for 'about:blank'; we map to binary only. | page-derived | medium |

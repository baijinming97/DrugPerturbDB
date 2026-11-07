options(stringsAsFactors=FALSE)
root<-'@WORKSPACE@/FULL_DATASET_FIGURES_V1'
e<-new.env();load(file.path(root,'raw/Exp_gzip.RData'),envir=e)
md<-read.delim(file.path(root,'raw/DRUGseq_U2OS_MoABox_plate_wells_metadata_public.txt'),check.names=FALSE,colClasses='character')
md<-md[md$well_type!='EMPTY',,drop=FALSE]
md<-md[order(md$plate_barcode,md$biosample_id),,drop=FALSE]
example<-e$Exp[[md$batch_id[1]]][[md$plate_barcode[1]]]
genes<-rownames(example$UMI.counts);stopifnot(!anyDuplicated(genes))
writeLines(genes,file.path(root,'data/DRUGseq_GENE_IDS.txt'))
write.table(md,file.path(root,'data/DRUGseq_STREAM_META.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
con<-file('/dev/stdout','wb');writeBin(as.integer(c(nrow(md),length(genes))),con,size=4,endian='little')
for(plate in unique(md$plate_barcode)){
 ii<-which(md$plate_barcode==plate);bb<-unique(md$batch_id[ii]);stopifnot(length(bb)==1)
 obj<-e$Exp[[bb]][[plate]];x<-obj$UMI.counts;stopifnot(setequal(rownames(x),genes));gi<-match(genes,rownames(x));si<-NULL
 for(k in c('biosample_id','external_biosample_id','plate_well','well_index','well_id'))if(all(md[[k]][ii]%in%colnames(x))){si<-match(md[[k]][ii],colnames(x));break}
 stopifnot(!is.null(si),!anyNA(si),!anyDuplicated(si));v<-x[gi,si,drop=FALSE];stopifnot(all(is.finite(v)),all(v>=0),all(v==round(v)),max(v)<.Machine$integer.max)
 writeBin(as.integer(length(ii)),con,size=4,endian='little');writeBin(as.integer(ii-1L),con,size=4,endian='little');writeBin(as.integer(v),con,size=4,endian='little');flush(con)
}
close(con)
